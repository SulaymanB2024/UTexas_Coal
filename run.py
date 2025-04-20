"""
Coal Price Forecasting Dashboard Runner
--------------------------------------
Script to launch the Coal Price Forecasting dashboard with proper
configuration and error handling.
"""
import os
import sys
import argparse
from dashboard.app import app, server

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Coal Price Forecasting Dashboard')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    return parser.parse_args()

def main():
    """Run the dashboard application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print startup information
    print("=" * 80)
    print("Coal Price Forecasting Dashboard")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Dashboard URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/")
    print("=" * 80)
    
    try:
        # Run the dashboard
        app.run_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running dashboard: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()