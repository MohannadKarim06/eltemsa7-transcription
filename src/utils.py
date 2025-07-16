import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_colab_environment():
    """Setup Google Colab environment"""
    logger.info("Setting up Google Colab environment...")

    # Install required packages
    packages = [
        "replicate",
        "requests"
    ]

    for package in packages:
        try:
            subprocess.run(["pip", "install", package], check=True, capture_output=True)
            logger.info(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        logger.info("✅ FFmpeg is available")
    except:
        logger.error("❌ FFmpeg not found. Please install it first.")
        return False

    logger.info("🎉 Environment setup complete!")
    return True