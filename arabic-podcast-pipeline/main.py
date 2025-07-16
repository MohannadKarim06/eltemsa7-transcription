import logging
import time
import json
import gc
import sys
import os
from pathlib import Path
from typing import List
from dataclasses import asdict

# Add the src directory to the Python path to allow relative imports
# This is a common pattern for making package imports work when running a script directly
# from the project root or a level above.
# Find the directory containing this script
script_dir = Path(__file__).parent.absolute()
src_path = script_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules from the src directory
try:
    import config
    from utils import setup_logging, setup_environment
    from audio_processor import AudioProcessor
    from file_uploader import FileUploader
    from transcriber import ReplicateTranscriber
    from speaker_labeler import SpeakerLabeler
    from data_models import Utterance # Assuming Utterance is needed in main for typing/saving
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print("and that the 'src' directory exists and contains all necessary files.")
    sys.exit(1)


# Configure logging early
setup_logging(config.LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class PodcastPipeline:
    """Main pipeline orchestrator with improved error handling"""

    def __init__(self):
        # Create directories
        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

        # Initialize output files
        self.episodes_file = Path(config.EPISODES_FILE)
        self.errors_file = Path(config.ERRORS_FILE)

        # Clear existing files
        if self.episodes_file.exists():
            # Only clear if we are starting fresh, might add resume logic later
            # For now, always clear for simplicity in refactoring
            try:
                self.episodes_file.unlink()
                logger.info(f"Cleared existing output file: {self.episodes_file}")
            except OSError as e:
                logger.warning(f"Could not clear existing output file {self.episodes_file}: {e}")


        if self.errors_file.exists():
            try:
                self.errors_file.unlink()
                logger.info(f"Cleared existing error log: {self.errors_file}")
            except OSError as e:
                 logger.warning(f"Could not clear existing error log {self.errors_file}: {e}")


        # Initialize components using configuration
        self.audio_processor = AudioProcessor(target_sr=config.TARGET_AUDIO_SR)
        self.file_uploader = FileUploader() # FileUploader might not need config passed explicitly if using hardcoded services
        self.transcriber = ReplicateTranscriber(
            hf_token=config.HF_TOKEN,
            replicate_token=config.REPLICATE_TOKEN,
            model_version=config.REPLICATE_MODEL_VERSION,
            base_url=config.REPLICATE_BASE_URL,
            blend_threshold=config.BLEND_THRESHOLD
        )
        self.speaker_labeler = SpeakerLabeler()

        logger.info("Pipeline initialized successfully")
        logger.info(f"Input directory: {config.INPUT_DIR}")
        logger.info(f"Output directory: {config.OUTPUT_DIR}")


    def process_episode(self, episode_path: Path) -> bool:
        """Process a single episode with comprehensive error handling"""
        episode_id = episode_path.stem
        logger.info(f"Processing episode: {episode_id}")

        # Ensure temp directory exists for this episode processing
        temp_episode_dir = Path(config.TEMP_DIR) # Use the global temp dir
        temp_episode_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

        try:
            # Stage 1: Preprocess audio
            temp_audio_path = temp_episode_dir / f"{episode_id}_clean.wav"

            logger.info(f"Stage 1: Preprocessing audio for {episode_id}")
            if not self.audio_processor.preprocess_audio(str(episode_path), str(temp_audio_path)):
                self._log_error(episode_id, "Audio preprocessing failed")
                return False

            # Stage 2: Upload audio to get public link
            logger.info(f"Stage 2: Uploading audio for {episode_id}")
            audio_url = self.file_uploader.upload_file(str(temp_audio_path), max_retries=config.UPLOADER_MAX_RETRIES)
            if not audio_url:
                self._log_error(episode_id, "Failed to upload audio")
                # Clean up temp file even on upload failure
                temp_audio_path.unlink(missing_ok=True)
                return False

            # Stage 3: Transcribe with speaker diarization
            logger.info(f"Stage 3: Transcribing {episode_id}")
            transcription_result = self.transcriber.transcribe_with_diarization(
                audio_url,
                max_retries=config.UPLOADER_MAX_RETRIES # Reuse uploader retries or define new? Let's reuse for now
            )

            # Clean up temp file after successful upload and before transcription
            temp_audio_path.unlink(missing_ok=True)
            logger.info(f"Cleaned up temporary audio file: {temp_audio_path}")

            if not transcription_result or not transcription_result.get("segments"):
                self._log_error(episode_id, "Transcription failed or returned no segments")
                return False

            # Stage 4: Label speakers and create utterances
            logger.info(f"Stage 4: Labeling speakers for {episode_id}")
            utterances = self.speaker_labeler.label_speakers(
                transcription_result["segments"], episode_id
            )

            if not utterances:
                self._log_error(episode_id, "No valid utterances created after labeling")
                return False

            # Stage 5: Save results
            logger.info(f"Stage 5: Saving results for {episode_id}")
            self._save_episode_results(utterances)

            logger.info(f"✅ Successfully processed {episode_id} - {len(utterances)} utterances")
            return True

        except Exception as e:
            error_msg = f"Error processing {episode_id}: {str(e)}"
            logger.error(error_msg)
            self._log_error(episode_id, error_msg)
            return False
        finally:
            # Ensure temp directory is cleaned up eventually if needed,
            # but keeping it per episode for simplicity might be okay for Colab.
            # If processing many files, cleaning per episode is better.
            # However, current setup uses a single temp dir.
            # Consider adding cleanup after each episode processing or a global cleanup.
            pass


    def _log_error(self, episode_id: str, error_message: str):
        """Log error to error file"""
        try:
            with open(self.errors_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{{timestamp}} - {{episode_id}}: {{error_message}}\n")
        except Exception as e:
            logger.error(f"Failed to write error to log file {self.errors_file}: {e}")


    def _save_episode_results(self, utterances: List[Utterance]):
        """Save episode results to episodes.jsonl file"""
        try:
            with open(self.episodes_file, 'a', encoding='utf-8') as f:
                for utterance in utterances:
                    # Ensure Utterance object is converted to a dictionary for JSON serialization
                    if isinstance(utterance, Utterance):
                         f.write(json.dumps(asdict(utterance), ensure_ascii=False) + '\n')
                    else:
                         # Handle cases where utterance might not be a Utterance object (shouldn't happen with current flow)
                         logger.warning(f"Attempted to save non-Utterance object: {{type(utterance)}}")
                         try:
                             f.write(json.dumps(utterance, ensure_ascii=False) + '\n')
                         except TypeError:
                              logger.error(f"Failed to serialize object to JSON: {{utterance}}")


        except Exception as e:
            logger.error(f"Failed to write results to output file {self.episodes_file}: {e}")


    def run_pipeline(self):
        """Run the complete pipeline on all episodes"""
        # Find all episode files
        input_path = Path(config.INPUT_DIR)
        if not input_path.exists():
             logger.error(f"Input directory does not exist: {config.INPUT_DIR}")
             return

        episode_files = []
        for pattern in ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac"]:
            episode_files.extend(input_path.glob(pattern))

        episode_files.sort()

        if not episode_files:
            logger.error(f"No audio files found in input directory: {config.INPUT_DIR}")
            return

        logger.info(f"Found {len(episode_files)} episodes to process")

        # Optional: Select a subset if needed for testing or resuming
        # episode_files = episode_files[103:] # Example subsetting

        # Process each episode
        success_count = 0
        start_time = time.time()

        for i, episode_file in enumerate(episode_files, 1):
            logger.info(f"\n{{'='*60}}")
            logger.info(f"Processing episode {{i}}/{{len(episode_files)}}: {{episode_file.name}}")
            logger.info(f"{{'='*60}}")

            if self.process_episode(episode_file):
                success_count += 1
            else:
                logger.error(f"❌ Failed to process: {{episode_file.name}}")

            # Log progress
            elapsed = time.time() - start_time
            # Avoid division by zero if i is 0 (shouldn't happen with enumerate(..., 1))
            avg_time = elapsed / i if i > 0 else 0
            remaining = (len(episode_files) - i) * avg_time if avg_time > 0 else 0

            logger.info(f"Progress: {{i}}/{{len(episode_files)}} | "
                       f"Success: {{success_count}} | "
                       f"Avg time: {{avg_time/60:.1f}}m/episode | "
                       f"ETA: {{remaining/60:.1f}}m")

            # Memory cleanup
            gc.collect()

        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Pipeline completed!")
        logger.info(f"📊 Results: {{success_count}}/{{len(episode_files)}} episodes processed successfully")
        logger.info(f"⏱️ Total time: {{total_time/60:.1f}} minutes")
        logger.info(f"📁 Output file: {{self.episodes_file}}")
        logger.info(f"📁 Error log: {{self.errors_file}}")

        # Show sample results
        self._show_sample_results()

    def _show_sample_results(self):
        """Show sample results from the output file"""
        if not self.episodes_file.exists():
            logger.info("\nNo output file found to display sample results.")
            return

        logger.info(f"\n📋 Sample results from {self.episodes_file.name}:")
        try:
            with open(self.episodes_file, 'r', encoding='utf-8') as f:
                lines = [next(f) for x in range(5)] # Read first 5 lines
            for i, line in enumerate(lines):
                try:
                    utterance = json.loads(line)
                    # Safely access keys with .get()
                    episode_id = utterance.get('episode_id', 'N/A')
                    speaker = utterance.get('speaker', 'N/A')
                    start = utterance.get('start', 0)
                    end = utterance.get('end', 0)
                    text = utterance.get('text', '')
                    logger.info(f"  {{episode_id}} | "
                               f"{{speaker}} | "
                               f"{{start:.1f}}s-{{end:.1f}}s | "
                               f"{{text[:50]}}...")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line {i+1} as JSON: {{line.strip()[:100]}}...")
                except Exception as e:
                     logger.warning(f"Error processing sample line {i+1}: {e}")

        except FileNotFoundError:
             logger.warning(f"Output file not found: {self.episodes_file}")
        except Exception as e:
            logger.error(f"Error reading sample results from file {self.episodes_file}: {e}")


def main():
    """Main function to run the pipeline"""
    logger.info("Starting pipeline execution...")

    # Setup environment (install packages, check ffmpeg)
    if not setup_environment():
        logger.error("Environment setup failed. Exiting.")
        sys.exit(1)

    # Configuration validation (basic checks)
    if config.HF_TOKEN == "your_huggingface_token_here" or config.REPLICATE_TOKEN == "your_replicate_token_here":
        logger.error("❌ Please update the HF_TOKEN and REPLICATE_TOKEN variables in config.py with your actual tokens")
        logger.info("Get your tokens from:")
        logger.info("  - HuggingFace: https://huggingface.co/settings/tokens")
        logger.info("  - Replicate: https://replicate.com/account/api-tokens")
        sys.exit(1)

    if not Path(config.INPUT_DIR).exists():
        logger.error(f"❌ Input directory does not exist: {config.INPUT_DIR}")
        logger.info("Please create the directory and add your audio files.")
        sys.exit(1)


    # Create and run pipeline
    try:
        pipeline = PodcastPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True) # Log traceback
        sys.exit(1)

    sys.exit(0) # Exit successfully

if __name__ == "__main__":
  # Ensure we are running from a context where the src directory is findable
  # The sys.path modification above helps when running directly from project root,
  # but if running from src, sys.path needs adjustment or run from root.
  # For typical execution (python main.py from project root), the above sys.path works.
  main()