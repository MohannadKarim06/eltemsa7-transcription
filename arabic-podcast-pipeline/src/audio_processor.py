import os
import subprocess
import logging
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio preprocessing and normalization"""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def preprocess_audio(self, input_path: str, output_path: str) -> bool:
        """Convert audio to mono WAV 16kHz and normalize"""
        try:
            # First, check if input file exists and is readable
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                return False

            # Get input file info
            try:
                probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if probe_result.returncode != 0:
                    logger.error(f"Cannot probe input file: {probe_result.stderr}")
                    return False

                probe_data = json.loads(probe_result.stdout)
                duration = float(probe_data['format']['duration'])
                logger.info(f"Input audio duration: {duration:.2f} seconds")

                # Skip very short files
                if duration < 10:
                    logger.warning(f"Audio file too short ({duration:.2f}s), skipping")
                    return False

            except Exception as e:
                logger.warning(f"Could not get audio duration: {e}")

            # Use ffmpeg for format conversion and normalization
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-ac', '1',  # Mono
                '-ar', str(self.target_sr),  # 16kHz
                '-acodec', 'pcm_s16le',  # Explicit codec
                '-filter:a', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Better normalization
                '-f', 'wav',  # Explicit format
                output_path
            ]

            logger.info(f"Converting audio: {input_path} -> {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            # Verify output file was created and is valid
            if not os.path.exists(output_path):
                logger.error(f"Output file was not created: {output_path}")
                return False

            # Check output file size
            output_size = os.path.getsize(output_path)
            if output_size < 1000:  # Less than 1KB is suspicious
                logger.error(f"Output file too small ({output_size} bytes)")
                return False

            logger.info(f"Audio preprocessed successfully: {output_path} ({output_size} bytes)")
            return True

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return False