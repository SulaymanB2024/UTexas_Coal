import logging
import yaml
import os
import sys

# Define the root directory of the project
# Assumes this script is in src/utils/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

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

def setup_logging():
  """Sets up the root logger based on configuration."""
  config = load_config()
  if config is None:
    print("Error: Could not load configuration for logging setup. Using default settings.", file=sys.stderr)
    log_level_str = 'INFO' # Default level
  else:
    log_level_str = config.get('project_settings', {}).get('log_level', 'INFO').upper()

  # Get the numeric level
  log_level = getattr(logging, log_level_str, logging.INFO)

  # Define the basic configuration for the root logger
  log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  date_format = '%Y-%m-%d %H:%M:%S'

  # Configure the root logger
  # Use force=True to allow reconfiguration if called multiple times (e.g., in notebooks)
  logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, stream=sys.stdout, force=True)

  # Optionally, quiet down overly verbose libraries
  # logging.getLogger('matplotlib').setLevel(logging.WARNING)
  # logging.getLogger('requests').setLevel(logging.WARNING)

  # Get the root logger and confirm setup
  root_logger = logging.getLogger()
  root_logger.info(f"Logging setup complete. Level set to: {log_level_str}")

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
  logger.debug("This is a debug message.") # Won't show if level is INFO
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
