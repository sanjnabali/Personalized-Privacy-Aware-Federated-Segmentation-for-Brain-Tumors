import yaml
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config(path="config.yaml"):
    """
    Loads the YAML config and injects env variables where needed.
    """
    # 1. Find the Project Root
    # This script is in src/utils/, so root is ../../
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # 2. Construct path to config.yaml
    config_path = os.path.join(project_root, path)

    if not os.path.exists(config_path):
        # Fallback: Try looking in current working directory
        if os.path.exists(path):
            config_path = path
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

    # 3. Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 4. Inject Private Key from .env safely
    private_key = os.getenv('GANACHE_PRIVATE_KEY')
    if private_key:
        config['blockchain']['private_key'] = private_key
    else:
        print("Warning: GANACHE_PRIVATE_KEY not found in .env")

    return config