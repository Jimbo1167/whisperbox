import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set up package-level logger
logger = logging.getLogger(__name__)

# Create a version variable
__version__ = "0.2.0"

# Log package initialization
logger.info(f"Whisperbox v{__version__} initialized")

# Create cache directory if it doesn't exist
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperbox")
os.makedirs(cache_dir, exist_ok=True)
