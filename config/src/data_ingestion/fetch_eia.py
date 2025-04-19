import requests
import pandas as pd
import os
import logging
from dotenv import load_dotenv
import sys

# Add project root to Python path to allow importing from src
# Assumes this script is in src/data_ingestion/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import setup_logging, load_config, CONFIG_PATH # Use absolute path for config

# Set up logging for this module
setup_logging()
logger = logging.getLogger(__name__)

def fetch_eia_data(series_id, api_key, config):
    """
    Fetches data for a specific series ID from the EIA API v2.

    Args:
        series_id (str): The EIA series ID to fetch (e.g., 'NG.RNGWHHD.D').
        api_key (str): Your EIA API key.
        config (dict): The loaded project configuration dictionary.

    Returns:
        pandas.DataFrame: DataFrame containing the fetched data (Date, Value),
                          or None if fetching fails.
    """
    if not api_key:
        logger.error("EIA API key not found. Please set EIA_API_KEY in your .env file.")
        return None
    if not series_id:
        logger.error("EIA Series ID not provided in config.")
        return None

    # Construct the API URL for EIA API v2
    # Reference: https://www.eia.gov/opendata/documentation.php
    base_url = config.get('data_sources', {}).get('eia', {}).get('api_url', 'https://api.eia.gov/v2/')
    # Ensure base_url ends with a slash
    if not base_url.endswith('/'):
        base_url += '/'

    # Parameters for the API request
    params = {
        'api_key': api_key,
        'facets[seriesId][]': series_id,
        'data[]': 'value',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000 # Max length per request for v2 API
    }
    # Note: For series with >5000 points, pagination logic would be needed here.
    # This implementation assumes fewer than 5000 data points for simplicity.

    endpoint = f'{series_id}/data/'
    full_url = base_url + endpoint

    logger.info(f"Requesting data for series '{series_id}' from EIA API V2...")
    logger.debug(f"Request URL (excluding api_key): {full_url}?facets[seriesId][]={series_id}&...") # Avoid logging key

    try:
        response = requests.get(full_url, params=params, timeout=30) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error during EIA API request: {e}")
        return None

    logger.info(f"API request successful (Status code: {response.status_code}). Parsing response...")

    try:
        data = response.json()

        # Check if the expected data structure is present
        if 'response' not in data or 'data' not in data['response']:
            logger.error(f"Unexpected API response structure: {data}")
            return None

        api_data = data['response']['data']

        if not api_data:
            logger.warning(f"No data returned from EIA API for series '{series_id}'.")
            return pd.DataFrame(columns=['Date', 'Value']) # Return empty DataFrame

        # Convert to DataFrame
        df = pd.DataFrame(api_data)

        # Select and rename relevant columns (adjust based on actual API response keys)
        # Common keys: 'period', 'value'. Verify these from API docs or response.
        if 'period' not in df.columns or 'value' not in df.columns:
             logger.error(f"Expected columns 'period' and 'value' not found in API response data keys: {df.columns}")
             return None

        df = df[['period', 'value']].copy()
        df.rename(columns={'period': 'Date', 'value': 'Value'}, inplace=True)

        # Convert 'Date' column based on frequency (heuristic: check format)
        # EIA V2 often uses YYYY-MM-DD for daily, YYYY-MM for monthly, YYYY for annual
        if df['Date'].iloc[0].count('-') == 2: # Daily or Monthly (YYYY-MM-DD or YYYY-MM)
            df['Date'] = pd.to_datetime(df['Date'])
        elif df['Date'].iloc[0].isdigit() and len(df['Date'].iloc[0]) == 4: # Annual (YYYY)
             df['Date'] = pd.to_datetime(df['Date'], format='%Y')
        else:
            logger.warning(f"Could not reliably determine date format for series '{series_id}'. Attempting standard parsing.")
            df['Date'] = pd.to_datetime(df['Date'])


        # Convert 'Value' column to numeric, coercing errors
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        # Handle potential missing values introduced by coercion
        if df['Value'].isnull().any():
            logger.warning(f"Some values could not be converted to numeric for series '{series_id}'. Check raw data.")

        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Successfully parsed {len(df)} data points for series '{series_id}'.")
        return df

    except ValueError as e: # Includes JSONDecodeError
        logger.error(f"Error parsing JSON response or converting data: {e}")
        return None
    except Exception as e: # Catch other potential errors during parsing/conversion
        logger.error(f"An unexpected error occurred during data processing: {e}")
        return None


def save_raw_data(df, series_id, config):
    """Saves the fetched data DataFrame to the raw data directory."""
    if df is None or df.empty:
        logger.warning(f"No data to save for series '{series_id}'.")
        return False

    try:
        raw_data_path = os.path.join(PROJECT_ROOT, config['data_paths']['raw'])
        os.makedirs(raw_data_path, exist_ok=True) # Ensure directory exists

        # Sanitize series_id for use in filename
        safe_series_id = series_id.replace('.', '_').replace('-', '_')
        filename = f"eia_{safe_series_id}_raw.csv"
        filepath = os.path.join(raw_data_path, filename)

        df.to_csv(filepath)
        logger.info(f"Successfully saved raw data for '{series_id}' to {filepath}")
        return True
    except KeyError:
        logger.error("Could not find 'data_paths.raw' in configuration.")
        return False
    except IOError as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during file saving: {e}")
        return False


# --- Main execution block ---
if __name__ == '__main__':
    logger.info("--- Starting EIA Data Fetching Script ---")

    # Load environment variables from .env file
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logger.info(".env file loaded.")
    else:
        logger.warning(".env file not found. API keys must be set as environment variables.")

    # Load configuration
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1) # Exit script if config fails

    # Get API Key from environment variable
    eia_api_key = os.getenv('EIA_API_KEY')
    if not eia_api_key:
         logger.warning("EIA_API_KEY not found in environment variables.")
         # Optionally exit or continue if API key isn't strictly needed for all runs
         # sys.exit(1)

    # Get Series ID from config
    try:
        # Example: Fetching Henry Hub data as defined in config
        henry_hub_series_id = config['data_sources']['eia']['henry_hub_series_id']
        logger.info(f"Target EIA series ID from config: {henry_hub_series_id}")
    except KeyError:
        logger.critical("Could not find 'data_sources.eia.henry_hub_series_id' in configuration. Exiting.")
        sys.exit(1)

    # Fetch the data
    eia_df = fetch_eia_data(henry_hub_series_id, eia_api_key, config)

    # Save the data
    if eia_df is not None:
        if save_raw_data(eia_df, henry_hub_series_id, config):
            logger.info("EIA data fetching and saving completed successfully.")
        else:
            logger.error("EIA data fetching completed, but saving failed.")
    else:
        logger.error("EIA data fetching failed.")

    logger.info("--- EIA Data Fetching Script Finished ---")

