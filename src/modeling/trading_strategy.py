"""Trading strategy implementation and economic value assessment for coal price forecasts."""
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Literal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Implements a trading strategy based on SARIMAX forecasts and evaluates its performance."""
    
    def __init__(
        self, 
        config: Dict,
        strategy_type: Literal["threshold", "directional", "both"] = "threshold",
        transaction_cost: float = 0.001,  # 0.1% of price
        slippage: float = 0.0005,         # 0.05% of price
        position_size: int = 1,           # 1 unit per trade
        threshold_sd_multiple: float = 0.5,  # Threshold as multiple of residual standard deviation
        directional_epsilon: float = 1e-6,  # Small value to avoid exact equality in directional strategy
        risk_free_rate: float = 0.02,     # 2% annual risk-free rate
        trading_frequency: str = 'ME'     # 'ME' for monthly end, 'D' for daily
    ):
        """Initialize the trading strategy.
        
        Args:
            config: Configuration dictionary
            strategy_type: Type of signal generation strategy to use ("threshold", "directional", or "both")
            transaction_cost: Transaction cost as a percentage of price
            slippage: Slippage as a percentage of price
            position_size: Number of units per trade
            threshold_sd_multiple: Threshold for trades as multiple of residual standard deviation
            directional_epsilon: Small value to avoid exact equality in directional strategy
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            trading_frequency: 'ME' for monthly end or 'D' for daily trading
        """
        self.config = config
        self.strategy_type = strategy_type
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_size = position_size
        self.threshold_sd_multiple = threshold_sd_multiple
        self.directional_epsilon = directional_epsilon
        self.risk_free_rate = risk_free_rate
        self.trading_frequency = trading_frequency
        
        # Will be set during calibration
        self.threshold = None
        self.residual_std = None
        
        # Results
        self.strategy_returns = None
        self.benchmark_returns = None
        self.strategy_metrics = None
        self.benchmark_metrics = None
        
    def calibrate_threshold(self, residuals: pd.Series) -> float:
        """Calibrate the strategy threshold based on residual standard deviation.
        
        Args:
            residuals: Model residuals from the training period
            
        Returns:
            Calibrated threshold value
        """
        self.residual_std = residuals.std()
        self.threshold = self.threshold_sd_multiple * self.residual_std
        
        logger.info(f"Calibrated strategy threshold: {self.threshold:.4f} "
                   f"({self.threshold_sd_multiple} × residual std of {self.residual_std:.4f})")
        
        return self.threshold
    
    def generate_signals(
        self, 
        actual_prices: pd.Series, 
        forecast_prices: pd.Series
    ) -> pd.DataFrame:
        """Generate trading signals based on forecasts and actual prices.
        
        Args:
            actual_prices: Series of actual prices
            forecast_prices: Series of forecasted prices
            
        Returns:
            DataFrame with position signals (1=long, -1=short, 0=flat)
        """
        if self.strategy_type in ["threshold", "both"] and self.threshold is None:
            raise ValueError("Threshold not calibrated. Call calibrate_threshold() first.")
            
        # Ensure data is aligned
        common_idx = actual_prices.index.intersection(forecast_prices.index)
        prices = actual_prices[common_idx]
        forecasts = forecast_prices[common_idx]
        
        # Calculate forecast-price difference
        diff = forecasts - prices
        
        # Initialize signals
        signals = pd.Series(0, index=common_idx)
        directional_signals = pd.Series(0, index=common_idx)
        threshold_signals = pd.Series(0, index=common_idx)
        
        # Generate directional signals based on forecast direction
        if self.strategy_type in ["directional", "both"]:
            # Long when forecast > price (with small epsilon to avoid exact equality)
            directional_signals[diff > self.directional_epsilon] = 1
            # Short when forecast < price
            directional_signals[diff < -self.directional_epsilon] = -1
            
            logger.info(f"Generated {len(directional_signals[directional_signals != 0])} directional signals "
                       f"({len(directional_signals[directional_signals > 0])} long, "
                       f"{len(directional_signals[directional_signals < 0])} short)")
        
        # Generate threshold-based signals
        if self.strategy_type in ["threshold", "both"]:
            # Long when forecast exceeds price by more than threshold
            threshold_signals[diff > self.threshold] = 1
            # Short when forecast is below price by more than threshold
            threshold_signals[diff < -self.threshold] = -1
            
            logger.info(f"Generated {len(threshold_signals[threshold_signals != 0])} threshold signals "
                       f"({len(threshold_signals[threshold_signals > 0])} long, "
                       f"{len(threshold_signals[threshold_signals < 0])} short)")
        
        # Assign final signals based on strategy type
        if self.strategy_type == "directional":
            signals = directional_signals
        elif self.strategy_type == "threshold":
            signals = threshold_signals
        elif self.strategy_type == "both":
            # Only take positions when both strategies agree
            signals = pd.Series(0, index=common_idx)
            signals[(directional_signals == 1) & (threshold_signals == 1)] = 1
            signals[(directional_signals == -1) & (threshold_signals == -1)] = -1
            
            logger.info(f"Generated {len(signals[signals != 0])} combined signals "
                       f"({len(signals[signals > 0])} long, "
                       f"{len(signals[signals < 0])} short)")
        
        # Create the trading dataframe
        trading_df = pd.DataFrame({
            'Price': prices,
            'Forecast': forecasts,
            'Diff': diff,
            'Signal': signals
        })
        
        # Add individual signal columns if using both strategies
        if self.strategy_type == "both":
            trading_df['Directional_Signal'] = directional_signals
            trading_df['Threshold_Signal'] = threshold_signals
        
        return trading_df
    
    def calculate_pnl(self, trading_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate profit and loss for the trading strategy.
        
        Args:
            trading_df: DataFrame with prices and signals
            
        Returns:
            DataFrame with additional P&L columns
        """
        # Copy input to avoid modifying the original
        result_df = trading_df.copy()
        
        # Calculate price returns
        result_df['Price_Return'] = result_df['Price'].pct_change()
        
        # Calculate position (with one-period lag to avoid look-ahead bias)
        result_df['Position'] = result_df['Signal'].shift(1)
        # Use first signal for first position to avoid losing a data point
        if len(result_df) > 0:
            result_df.loc[result_df.index[0], 'Position'] = 0
        
        # Identify trades (position changes)
        result_df['Trade'] = result_df['Position'].diff()
        # First trade is the initial position
        if len(result_df) > 0:
            result_df.loc[result_df.index[0], 'Trade'] = result_df.loc[result_df.index[0], 'Position']
        
        # Calculate number of trades
        num_trades = (result_df['Trade'] != 0).sum()
        logger.info(f"Total number of trades executed: {num_trades}")
        
        # Calculate transaction costs
        result_df['Transaction_Cost'] = 0.0
        # Apply costs only when there's a trade
        trade_mask = result_df['Trade'] != 0
        if trade_mask.any():
            result_df.loc[trade_mask, 'Transaction_Cost'] = (
                abs(result_df.loc[trade_mask, 'Trade']) * 
                result_df.loc[trade_mask, 'Price'] * 
                (self.transaction_cost + self.slippage) * 
                self.position_size
            )
        
        # Calculate P&L (without transaction costs)
        result_df['Strategy_Return'] = (
            result_df['Position'] * 
            result_df['Price_Return'] * 
            self.position_size
        )
        
        # Apply transaction costs
        result_df['Net_Return'] = (
            result_df['Strategy_Return'] - 
            result_df['Transaction_Cost'] / result_df['Price']
        )
        
        # Calculate cumulative returns
        result_df['Cum_Strategy_Return'] = (1 + result_df['Net_Return']).cumprod()
        result_df['Cum_Price_Return'] = (1 + result_df['Price_Return']).cumprod()
        
        # Calculate running drawdowns
        peak = result_df['Cum_Strategy_Return'].expanding().max()
        result_df['Drawdown'] = (result_df['Cum_Strategy_Return'] / peak) - 1
        
        return result_df
    
    def calculate_performance_metrics(
        self, 
        returns: pd.Series, 
        name: str = 'Strategy'
    ) -> Dict[str, float]:
        """Calculate performance metrics for a return series.
        
        Args:
            returns: Series of period returns
            name: Name of the strategy for logging
            
        Returns:
            Dictionary of performance metrics
        """
        # Remove NaN values
        returns = returns.dropna()
        
        # If no valid returns, return zeros for all metrics
        if len(returns) == 0:
            metrics = {
                'Total_Return': 0.0,
                'Annualized_Return': 0.0,
                'Annualized_Volatility': 0.0,
                'Sharpe_Ratio': 0.0,
                'Sortino_Ratio': 0.0,
                'Max_Drawdown': 0.0,
                'Win_Rate': 0.0,
                'Total_Trades': 0,
                'Profit_Factor': 0.0,
                'Avg_Return_Per_Trade': 0.0
            }
            logger.warning(f"{name} has no valid returns for performance calculation")
            return metrics
        
        # Determine annualization factor based on trading frequency
        if self.trading_frequency == 'D':
            ann_factor = 252
        elif self.trading_frequency in ['M', 'ME']:
            ann_factor = 12
        else:
            ann_factor = 12  # Default to monthly
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(ann_factor)
        
        # Sharpe ratio
        if ann_vol > 0:
            sharpe = (ann_return - self.risk_free_rate) / ann_vol
        else:
            sharpe = 0
        
        # Sortino ratio (using downside volatility)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 0
        sortino = (ann_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        # Total number of trades (approximated from return series)
        non_zero_returns = returns[returns != 0]
        total_trades = len(non_zero_returns)
        
        # Profit factor (sum of positive returns / abs sum of negative returns)
        sum_wins = returns[returns > 0].sum()
        sum_losses = abs(returns[returns < 0].sum())
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else float('inf')
        
        # Average return per trade
        avg_return_per_trade = returns.mean() if total_trades > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            'Total_Return': total_return,
            'Annualized_Return': ann_return,
            'Annualized_Volatility': ann_vol,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Trades': total_trades,
            'Profit_Factor': profit_factor,
            'Avg_Return_Per_Trade': avg_return_per_trade
        }
        
        # Log metrics
        logger.info(f"\n{name} Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
            
        return metrics
    
    def evaluate_strategy(
        self, 
        train_residuals: pd.Series,
        actual_prices: pd.Series, 
        forecast_prices: pd.Series
    ) -> Dict[str, Union[Dict[str, float], pd.DataFrame]]:
        """Evaluate trading strategy performance compared to a buy-and-hold benchmark.
        
        Args:
            train_residuals: Residuals from training period for threshold calibration
            actual_prices: Series of actual prices
            forecast_prices: Series of forecasted prices
            
        Returns:
            Dictionary containing performance metrics and trading dataframe
        """
        # Calibrate threshold if using threshold-based strategies
        if self.strategy_type in ["threshold", "both"]:
            self.calibrate_threshold(train_residuals)
        
        # Generate signals
        trading_df = self.generate_signals(actual_prices, forecast_prices)
        
        # Calculate strategy P&L
        result_df = self.calculate_pnl(trading_df)
        
        # Calculate metrics for strategy
        self.strategy_returns = result_df['Net_Return']
        self.strategy_metrics = self.calculate_performance_metrics(
            self.strategy_returns, 
            name=f'Trading Strategy ({self.strategy_type})'
        )
        
        # Calculate metrics for buy-and-hold benchmark
        self.benchmark_returns = result_df['Price_Return']
        self.benchmark_metrics = self.calculate_performance_metrics(
            self.benchmark_returns, 
            name='Buy-and-Hold Benchmark'
        )
        
        # Compare strategy vs benchmark
        logger.info("\nStrategy vs Benchmark Comparison:")
        for metric in self.strategy_metrics.keys():
            strategy_value = self.strategy_metrics[metric]
            benchmark_value = self.benchmark_metrics.get(metric, 0)
            
            if isinstance(strategy_value, float) and isinstance(benchmark_value, float):
                diff = strategy_value - benchmark_value
                logger.info(f"{metric}: Strategy {strategy_value:.4f} vs Benchmark {benchmark_value:.4f}"
                          f" (Diff: {diff:.4f})")
            else:
                logger.info(f"{metric}: Strategy {strategy_value} vs Benchmark {benchmark_value}")
        
        return {
            'strategy_metrics': self.strategy_metrics,
            'benchmark_metrics': self.benchmark_metrics,
            'trading_df': result_df
        }

    def plot_performance(self, result_df: pd.DataFrame, output_path: str = None) -> None:
        """Plot trading strategy performance vs benchmark.
        
        Args:
            result_df: DataFrame with trading results
            output_path: Path to save the figure
        """
        fig = make_subplots(
            rows=4, 
            cols=1,
            subplot_titles=(
                'Cumulative Returns',
                'Trading Positions',
                'Monthly Returns Comparison',
                'Drawdown'
            ),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Plot cumulative returns
        fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=result_df['Cum_Strategy_Return'],
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=result_df['Cum_Price_Return'],
                mode='lines',
                name='Buy-and-Hold',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Add annotations for key metrics
        strategy_return = result_df['Cum_Strategy_Return'].iloc[-1] if len(result_df) > 0 else 1
        benchmark_return = result_df['Cum_Price_Return'].iloc[-1] if len(result_df) > 0 else 1
        
        # Plot signals and positions
        fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=result_df['Position'],
                mode='lines',
                name='Position',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Plot signals as markers
        long_signals = result_df[result_df['Signal'] == 1]
        short_signals = result_df[result_df['Signal'] == -1]
        
        if not long_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_signals.index,
                    y=[1] * len(long_signals),
                    mode='markers',
                    name='Long Signal',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ),
                row=2, col=1
            )
        
        if not short_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_signals.index,
                    y=[-1] * len(short_signals),
                    mode='markers',
                    name='Short Signal',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ),
                row=2, col=1
            )
        
        # Create monthly return comparison bars
        monthly_strategy = result_df['Net_Return'].resample('ME').sum()
        monthly_benchmark = result_df['Price_Return'].resample('ME').sum()
        
        # Plot monthly returns
        fig.add_trace(
            go.Bar(
                x=monthly_strategy.index,
                y=monthly_strategy,
                name='Strategy Monthly Return',
                marker_color='blue'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_benchmark.index,
                y=monthly_benchmark,
                name='Benchmark Monthly Return',
                marker_color='green',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # Plot drawdown
        fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=result_df['Drawdown'],
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=4, col=1
        )
        
        # Add horizontal line at maximum drawdown
        max_dd = result_df['Drawdown'].min() if len(result_df) > 0 else 0
        fig.add_shape(
            type="line",
            x0=result_df.index.min(),
            y0=max_dd,
            x1=result_df.index.max(),
            y1=max_dd,
            line=dict(color="red", width=1, dash="dash"),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Trading Strategy ({self.strategy_type}) Performance vs Buy-and-Hold Benchmark',
            height=1000,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=[
                dict(
                    x=0.01,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    text=f"Strategy Return: {(strategy_return - 1) * 100:.2f}% | Buy-Hold: {(benchmark_return - 1) * 100:.2f}%<br>Max DD: {max_dd * 100:.2f}% | Sharpe: {self.strategy_metrics.get('Sharpe_Ratio', 0):.2f}",
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        # Add y-axis ranges
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Position", range=[-1.2, 1.2], row=2, col=1)
        fig.update_yaxes(title_text="Monthly Return", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown", row=4, col=1)
        
        # Save figure if path provided
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Performance plot saved to {output_path}")
        
        return fig
    
    def save_results(self, result_df: pd.DataFrame, output_file: str) -> None:
        """Save trading strategy results to CSV.
        
        Args:
            result_df: DataFrame with trading results
            output_file: Path to save the results
        """
        # Create a summary DataFrame
        metrics_df = pd.DataFrame({
            'Metric': list(self.strategy_metrics.keys()),
            'Strategy': list(self.strategy_metrics.values()),
            'Benchmark': [self.benchmark_metrics.get(k, 0) for k in self.strategy_metrics.keys()],
            'Difference': [
                self.strategy_metrics[k] - self.benchmark_metrics.get(k, 0)
                for k in self.strategy_metrics.keys()
            ]
        })
        
        # Add strategy type and parameters
        params_df = pd.DataFrame({
            'Metric': ['Strategy_Type', 'Transaction_Cost', 'Slippage', 'Position_Size', 
                      'Risk_Free_Rate', 'Threshold_SD_Multiple'],
            'Strategy': [self.strategy_type, self.transaction_cost, self.slippage, 
                        self.position_size, self.risk_free_rate, self.threshold_sd_multiple],
            'Benchmark': [None, None, None, None, None, None],
            'Difference': [None, None, None, None, None, None]
        })
        
        # Combine metrics and parameters
        final_df = pd.concat([metrics_df, params_df], ignore_index=True)
        
        # Save metrics to CSV
        final_df.to_csv(output_file, index=False)
        logger.info(f"Economic value results saved to {output_file}")
        
        # Save detailed results
        detailed_output = output_file.replace('.csv', '_detailed.csv')
        result_df.to_csv(detailed_output)
        logger.info(f"Detailed trading results saved to {detailed_output}")
        
        # Create trade log for analysis
        trade_log = self._create_trade_log(result_df)
        trade_log_output = output_file.replace('.csv', '_trades.csv')
        if trade_log is not None:
            trade_log.to_csv(trade_log_output)
            logger.info(f"Trade log saved to {trade_log_output}")
    
    def _create_trade_log(self, result_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create a log of individual trades for analysis.
        
        Args:
            result_df: DataFrame with trading results
            
        Returns:
            DataFrame containing trade details or None if no trades
        """
        # Identify trades (position changes)
        trades = result_df[result_df['Trade'] != 0].copy()
        
        if trades.empty:
            logger.info("No trades executed during the period")
            return None
        
        # Calculate metrics for each trade
        trades['Direction'] = trades['Trade'].apply(lambda x: 'Long' if x > 0 else 'Short')
        trades['Entry_Price'] = trades['Price']
        
        # For each trade, find exit point
        trade_records = []
        
        for i, trade in enumerate(trades.itertuples()):
            trade_dict = {
                'Entry_Date': trade.Index,
                'Entry_Price': trade.Price,
                'Position': trade.Trade,
                'Direction': trade.Direction,
                'Transaction_Cost': trade.Transaction_Cost
            }
            
            # Find exit point (next trade or end of dataset)
            if i < len(trades) - 1:
                exit_row = trades.iloc[i+1]
                trade_dict['Exit_Date'] = exit_row.name
                trade_dict['Exit_Price'] = exit_row['Price']
                trade_dict['Holding_Period'] = (exit_row.name - trade.Index).days
            else:
                # Last trade - exit at the end of the dataset
                exit_row = result_df.iloc[-1]
                trade_dict['Exit_Date'] = exit_row.name
                trade_dict['Exit_Price'] = exit_row['Price']
                trade_dict['Holding_Period'] = (exit_row.name - trade.Index).days
            
            # Calculate P&L
            if trade.Direction == 'Long':
                trade_dict['PnL'] = (trade_dict['Exit_Price'] - trade_dict['Entry_Price']) * abs(trade.Trade) - trade.Transaction_Cost
                trade_dict['PnL_Pct'] = (trade_dict['Exit_Price'] / trade_dict['Entry_Price'] - 1) * 100
            else:  # Short
                trade_dict['PnL'] = (trade_dict['Entry_Price'] - trade_dict['Exit_Price']) * abs(trade.Trade) - trade.Transaction_Cost
                trade_dict['PnL_Pct'] = (trade_dict['Entry_Price'] / trade_dict['Exit_Price'] - 1) * 100
            
            trade_records.append(trade_dict)
        
        # Create DataFrame
        trade_log = pd.DataFrame(trade_records)
        
        # Add summary stats
        if not trade_log.empty:
            win_count = len(trade_log[trade_log['PnL'] > 0])
            loss_count = len(trade_log[trade_log['PnL'] <= 0])
            win_rate = win_count / len(trade_log) if len(trade_log) > 0 else 0
            
            logger.info(f"Trade Statistics:")
            logger.info(f"Total Trades: {len(trade_log)}")
            logger.info(f"Winning Trades: {win_count} ({win_rate:.2%})")
            logger.info(f"Losing Trades: {loss_count} ({1-win_rate:.2%})")
            logger.info(f"Average Holding Period: {trade_log['Holding_Period'].mean():.1f} days")
            logger.info(f"Average P&L per trade: {trade_log['PnL'].mean():.2f}")
            
        return trade_log