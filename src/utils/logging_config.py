import logging
import yaml
import os
import sys
from typing import Dict

# Define the root directory of the project
# Assumes this script is in src/utils/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

# Global flag to track if logging has been set up
_LOGGING_SETUP_DONE = False

def load_config(config_path=CONFIG_PATH):
    """Loads the YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The loaded configuration dictionary.
        None: If the file cannot be found or parsed.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}", file=sys.stderr)
        return None

def setup_logging(config: Dict = None) -> None:
    """Setup logging configuration.
    
    Args:
        config: Optional configuration dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete. Level set to: INFO")

# --- Example Usage (can be removed or kept for testing) ---
if __name__ == '__main__':
    # This block runs only when the script is executed directly
    print(f"Project Root Directory: {PROJECT_ROOT}")
    print(f"Config Path: {CONFIG_PATH}")

    # Set up logging
    setup_logging()

    # Get a logger for this specific module
    logger = logging.getLogger(__name__)

    # Log some messages at different levels
    logger.debug("This is a debug message.")  # Won't show if level is INFO
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # Test loading config directly
    cfg = load_config()
    if cfg:
        print("\nSuccessfully loaded configuration snippet:")
        print(f"  Target Variable: {cfg.get('variables', {}).get('target_variable')}")
        print(f"  Log Level Setting: {cfg.get('project_settings', {}).get('log_level')}")
