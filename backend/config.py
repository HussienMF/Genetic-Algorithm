# import logging
# from pathlib import Path

# # Define paths
# PROJECT_ROOT = Path(__file__).parent.parent
# UPLOAD_DIR = PROJECT_ROOT / "uploads"
# OUTPUT_BASE = PROJECT_ROOT / "outputs"

# # Ensure directories exist
# UPLOAD_DIR.mkdir(exist_ok=True)
# OUTPUT_BASE.mkdir(exist_ok=True)

# # Configure logging
# # If running inside Docker use /app/logs, otherwise use a local logs directory
# docker_log_dir = Path('/app/logs')
# if docker_log_dir.exists():
#     log_dir = docker_log_dir
# else:
#     log_dir = PROJECT_ROOT / 'logs'
#     log_dir.mkdir(parents=True, exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(str(log_dir / 'app.log')),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # CORS configuration (can be moved to .env later)
# ALLOWED_ORIGINS = ["*"]
# import logging
# from pathlib import Path

# # Define paths
# PROJECT_ROOT = Path(__file__).parent.parent
# UPLOAD_DIR = PROJECT_ROOT / "uploads"
# OUTPUT_BASE = PROJECT_ROOT / "outputs"

# # Ensure directories exist
# UPLOAD_DIR.mkdir(exist_ok=True)
# OUTPUT_BASE.mkdir(exist_ok=True)

# # Configure logging
# log_dir = PROJECT_ROOT / "logs"
# log_dir.mkdir(exist_ok=True)
# log_file = log_dir / "app.log"

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(str(log_file)),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # CORS configuration (can be moved to .env later)
# ALLOWED_ORIGINS = ["*"]


import logging
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_BASE = PROJECT_ROOT / "outputs"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_BASE.mkdir(exist_ok=True)

# Configure logging for Docker or local
docker_log_dir = Path('/app/logs')
if docker_log_dir.exists():
    log_dir = docker_log_dir
else:
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CORS configuration
ALLOWED_ORIGINS = ["*"]