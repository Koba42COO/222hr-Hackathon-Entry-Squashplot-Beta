"""Basic logging configuration for the project"""

import logging
import logging.handlers
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file="app.log"):
    """Setup comprehensive logging configuration"""

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Create logger
    logger = logging.getLogger(__name__)
    return logger

# Default logger instance
logger = setup_logging()
