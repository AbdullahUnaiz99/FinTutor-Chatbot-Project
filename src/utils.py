import os
from dotenv import load_dotenv

def ensure_data_folder():
    """Ensure data folder exists"""
    os.makedirs("Data", exist_ok=True)
    print("Data folder ensured to exist")

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["PINECONE_API_KEY", "OPEN_ROUTER_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("Environment variables loaded successfully")

def validate_file_path(file_path):
    """Validate if file path exists"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

