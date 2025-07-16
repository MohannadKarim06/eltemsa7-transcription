import os

# Root directory of the project
ROOT_DIR = "arabic-podcast-pipeline"

# Input and Output Directories
# Default paths assuming execution in Google Colab
INPUT_DIR = os.path.join(ROOT_DIR, "/content/drive/MyDrive/Untitled Folder/downloaded_audio")
OUTPUT_DIR = os.path.join(ROOT_DIR, "/content/drive/MyDrive/Untitled Folder/processed")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp") # Temporary directory for processed audio


# API Tokens
# IMPORTANT: Replace these with your actual tokens
# Get tokens from:
#   - HuggingFace: https://huggingface.co/settings/tokens
#   - Replicate: https://replicate.com/account/api-tokens
HF_TOKEN = "your_huggingface_token_here"
REPLICATE_TOKEN = "your_replicate_token_here"

# Replicate Model Configuration
REPLICATE_MODEL_VERSION = "84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
REPLICATE_BASE_URL = "https://api.replicate.com/v1/predictions"
REPLICATE_TIMEOUT = 1200 # Timeout for Replicate prediction request in seconds

# Audio Processing Configuration
TARGET_AUDIO_SR = 16000 # Target sample rate for audio processing

# Transcription and Diarization Configuration
BLEND_THRESHOLD = 2.0 # Max time gap (seconds) between utterances to blend

# File Uploader Configuration
UPLOADER_MAX_RETRIES = 3 # Max retries for file uploads

# Polling Configuration
POLLING_MAX_WAIT_TIME = 1800 # Max time to wait for Replicate prediction completion
POLLING_INTERVAL = 15 # Initial polling interval in seconds

# Logging Configuration
LOGGING_LEVEL = "INFO" # Options: "DEBUG", "INFO", "WARNING", "ERROR"

# Other constants
EPISODES_FILE = os.path.join(OUTPUT_DIR, "episodes.jsonl")
ERRORS_FILE = os.path.join(OUTPUT_DIR, "errors.log")