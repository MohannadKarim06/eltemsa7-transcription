import logging
import subprocess
import os
from pathlib import Path
import sys

def setup_logging(log_level: str = "INFO"):
    """
    Configures basic logging for the application.

    Args:
        log_level: The desired logging level (e.g., "DEBUG", "INFO", "WARNING").
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {logging.getLevelName(level)}")

def setup_environment():
    """
    Handles environment setup, including installing necessary packages
    and checking for dependencies like ffmpeg.

    Returns:
        True if setup is successful, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up environment...")

    # Install required packages - read from requirements.txt
    try:
        # Find the project root directory by looking for a known marker, e.g., 'requirements.txt'
        current_dir = Path(os.getcwd())
        project_root = None
        # Search up to 5 levels up
        for _ in range(5):
            if (current_dir / "requirements.txt").exists():
                project_root = current_dir
                break
            if current_dir.parent == current_dir: # Reached root filesystem
                break
            current_dir = current_dir.parent

        if project_root:
            requirements_path = project_root / "requirements.txt"
            logger.info(f"Installing packages from {{requirements_path}}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True, capture_output=True)
            logger.info("✅ Installed required packages")
        else:
             logger.warning("requirements.txt not found. Attempting to install known packages directly.")
             # Attempt to install known packages directly if requirements.txt is missing
             packages = ["yt-dlp", "replicate", "requests"]
             subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True, capture_output=True)
             logger.info(f"✅ Installed packages directly: {{', '.join(packages)}}")


    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install packages: {{e.stderr.decode()}}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during package installation: {{e}}")
        return False


    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        logger.info("✅ FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ FFmpeg not found. Please install it first.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while checking FFmpeg: {{e}}")
        return False


    logger.info("🎉 Environment setup complete!")
    return True

# The example usage block (__main__ check) should be removed
# as this file is intended to be imported as a module.