import logging
import time
import gc
import json
from pathlib import Path
from typing import List

# Assuming these modules are in the src directory
from src.audio_downloader import AudioDownloader
from src.audio_processor import AudioProcessor
from src.file_uploader import FileUploader
from src.transcriber import ReplicateTranscriber
from src.speaker_labeler import SpeakerLabeler
from src.utils import setup_colab_environment # Import setup_colab_environment from utils
from src.data_models import Utterance # Import Utterance dataclass

# Import configuration
try:
    from config import INPUT_DIR, OUTPUT_DIR, HF_TOKEN, REPLICATE_TOKEN
except ImportError:
    logging.error("Could not import configuration from config.py. Please ensure it exists and is in the same directory.")
    # Set dummy values or exit if config is essential
    INPUT_DIR = "."
    OUTPUT_DIR = "./processed_output"
    HF_TOKEN = "dummy_hf_token"
    REPLICATE_TOKEN = "dummy_replicate_token"


# Configure logging (consistent with other modules)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PodcastPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, input_dir: str, output_dir: str, hf_token: str, replicate_token: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp" # Use a temp directory within output

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True) # Create temp directory

        # Initialize output files
        self.episodes_file = self.output_dir / "episodes.jsonl"
        self.errors_file = self.output_dir / "errors.log"

        # Clear existing files (optional, depending on desired behavior)
        # if self.episodes_file.exists():
        #     self.episodes_file.unlink()
        # if self.errors_file.exists():
        #     self.errors_file.unlink()
        # logger.info("Cleared previous output files.")


        # Initialize components
        # AudioDownloader is not used here, it's assumed audio is already in INPUT_DIR
        self.audio_processor = AudioProcessor()
        self.file_uploader = FileUploader()
        # Use the blend_threshold from config or a default
        self.transcriber = ReplicateTranscriber(hf_token, replicate_token, blend_threshold=2.0)
        self.speaker_labeler = SpeakerLabeler()

        logger.info("PodcastPipeline initialized successfully")

    def process_episode(self, episode_path: Path) -> bool:
        """Process a single episode with comprehensive error handling"""
        # Ensure episode_path is a Path object
        episode_path = Path(episode_path)
        episode_id = episode_path.stem
        logger.info(f"Processing episode: {episode_id}")

        try:
            # Stage 1: Preprocess audio
            temp_audio_path = self.temp_dir / f"{episode_id}_clean.wav"

            logger.info(f"Stage 1: Preprocessing audio for {episode_id} from {episode_path}")
            if not self.audio_processor.preprocess_audio(str(episode_path), str(temp_audio_path)):
                self._log_error(episode_id, "Audio preprocessing failed")
                # Clean up temp file even on failure
                temp_audio_path.unlink(missing_ok=True)
                return False

            # Stage 2: Upload audio to get public link
            logger.info(f"Stage 2: Uploading audio for {episode_id} from {temp_audio_path}")
            audio_url = self.file_uploader.upload_file(str(temp_audio_path))
            # Clean up temp file immediately after upload attempt
            temp_audio_path.unlink(missing_ok=True)
            if not audio_url:
                self._log_error(episode_id, "Failed to upload audio")
                return False

            # Stage 3: Transcribe with speaker diarization
            logger.info(f"Stage 3: Transcribing {episode_id} from {audio_url}")
            transcription_result = self.transcriber.transcribe_with_diarization(audio_url)

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
            logger.info(f"Stage 5: Saving results for {episode_id} ({len(utterances)} utterances)")
            self._save_episode_results(utterances)

            logger.info(f"✅ Successfully processed {episode_id} - {len(utterances)} utterances saved")
            return True

        except Exception as e:
            error_msg = f"Unexpected error processing {episode_id}: {str(e)}"
            logger.error(error_msg, exc_info=True) # Log traceback
            self._log_error(episode_id, error_msg)
            # Ensure temp file is cleaned up even on unexpected errors
            temp_audio_path = self.temp_dir / f"{episode_id}_clean.wav"
            temp_audio_path.unlink(missing_ok=True)
            return False

    def _log_error(self, episode_id: str, error_message: str):
        """Log error to error file"""
        try:
            from datetime import datetime
            with open(self.errors_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} - {episode_id}: {error_message}\n")
            logger.debug(f"Logged error for {episode_id} to {self.errors_file}")
        except Exception as e:
            logger.error(f"Failed to write error for {episode_id} to error log file: {e}")


    def _save_episode_results(self, utterances: List[Utterance]):
        """Save episode results to episodes.jsonl file"""
        try:
            with open(self.episodes_file, 'a', encoding='utf-8') as f:
                for utterance in utterances:
                    # Ensure Utterance object is converted to dict before dumping
                    f.write(json.dumps(asdict(utterance), ensure_ascii=False) + '\n')
            logger.debug(f"Saved {len(utterances)} utterances to {self.episodes_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {self.episodes_file}: {e}")


    def run_pipeline(self):
        """Run the complete pipeline on all episodes"""
        # Find all episode files
        episode_files = []
        # Supported audio extensions
        supported_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac", "*.ogg"]
        for pattern in supported_extensions:
            episode_files.extend(self.input_dir.glob(pattern))

        episode_files.sort()

        if not episode_files:
            logger.error(f"No audio files found in input directory: {self.input_dir}")
            logger.info("Please ensure audio files are in the specified INPUT_DIR in config.py")
            return

        logger.info(f"Found {len(episode_files)} episodes to process in {self.input_dir}")

        # You might want to process a subset for testing, e.g., episode_files[:10]
        # episode_files_to_process = episode_files # Process all
        # episode_files_to_process = episode_files[103:] # Continue from a specific index
        # episode_files_to_process = episode_files[:5] # Process first 5

        # For this subtask, we will iterate through all found files as per the instruction's intent
        episode_files_to_process = episode_files


        # Process each episode
        success_count = 0
        start_time = time.time()

        for i, episode_file in enumerate(episode_files_to_process, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing episode {i}/{len(episode_files_to_process)}: {episode_file.name}")
            logger.info(f"{'='*60}")

            if self.process_episode(episode_file):
                success_count += 1
            else:
                logger.error(f"❌ Failed to process: {episode_file.name}")

            # Log progress
            elapsed = time.time() - start_time
            avg_time = elapsed / i if i > 0 else 0
            remaining = (len(episode_files_to_process) - i) * avg_time if avg_time > 0 else 0

            logger.info(f"Progress: {i}/{len(episode_files_to_process)} | "
                       f"Success: {success_count} | "
                       f"Avg time: {avg_time/60:.1f}m/episode | "
                       f"ETA: {remaining/60:.1f}m")

            # Memory cleanup
            gc.collect()
            logger.debug("Ran garbage collection.")


        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Pipeline completed!")
        logger.info(f"📊 Results: {success_count}/{len(episode_files_to_process)} episodes processed successfully")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Output file: {self.episodes_file}")
        logger.info(f"📁 Error log: {self.errors_file}")

        # Show sample results
        self._show_sample_results()

    def _show_sample_results(self):
        """Show sample results from the output file"""
        if not self.episodes_file.exists():
            logger.info(f"Output file {self.episodes_file.name} does not exist yet.")
            return

        logger.info(f"\n📋 Sample results from {self.episodes_file.name}:")
        try:
            with open(self.episodes_file, 'r', encoding='utf-8') as f:
                lines = [next(f) for x in range(5) if x < sum(1 for line in open(self.episodes_file))] # Read up to 5 lines safely

            if not lines:
                 logger.info("Output file is empty.")
                 return

            for line in lines:
                try:
                    utterance = json.loads(line)
                    # Format the output nicely
                    episode_id = utterance.get('episode_id', 'N/A')
                    speaker = utterance.get('speaker', 'N/A')
                    start = utterance.get('start', 0.0)
                    end = utterance.get('end', 0.0)
                    text = utterance.get('text', '').strip()

                    logger.info(f"  Episode: {episode_id} | "
                               f"Speaker: {speaker:<5} | " # Pad speaker label
                               f"Time: {start:.1f}s-{end:.1f}s | "
                               f"Text: {text[:80]}...") # Show first 80 chars
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line as JSON: {line.strip()[:100]}...")
                except Exception as e:
                    logger.warning(f"Error processing line from output file: {e} - {line.strip()[:100]}...")

        except FileNotFoundError:
             logger.error(f"Output file {self.episodes_file.name} not found.")
        except Exception as e:
            logger.error(f"An error occurred while reading sample results: {e}")


def main():
    """Main function to run the pipeline"""

    # Setup environment (Colab specific)
    if not setup_colab_environment():
        logger.error("Environment setup failed. Exiting.")
        return

    # Configuration is imported from config.py
    # Validate tokens and input directory
    if HF_TOKEN == "your_huggingface_token_here" or REPLICATE_TOKEN == "your_replicate_token_here":
        logger.error("❌ Please update the HF_TOKEN and REPLICATE_TOKEN variables in config.py with your actual tokens")
        logger.info("Get your tokens from:")
        logger.info("  - HuggingFace: https://huggingface.co/settings/tokens")
        logger.info("  - Replicate: https://replicate.com/account/api-tokens")
        return

    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        logger.error(f"❌ Input directory does not exist: {INPUT_DIR}")
        logger.info("Please create the directory and add your audio files, or update INPUT_DIR in config.py")
        return

    if not input_path.is_dir():
        logger.error(f"❌ Input path is not a directory: {INPUT_DIR}")
        return


    # Create and run pipeline
    try:
        logger.info("Starting Podcast Transcription Pipeline...")
        pipeline = PodcastPipeline(INPUT_DIR, OUTPUT_DIR, HF_TOKEN, REPLICATE_TOKEN)
        pipeline.run_pipeline()
        logger.info("Podcast Transcription Pipeline finished.")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True) # Log traceback
        # Decide if you want to re-raise the exception or just log and exit
        # raise # Uncomment to re-raise


if __name__ == "__main__":
  main()