import logging
import time
import gc
import json
from pathlib import Path
from typing import List
from dataclasses import asdict

# Assuming these modules are in the src directory
from src.audio_downloader import AudioDownloader
from src.audio_processor import AudioProcessor
from src.file_uploader import FileUploader
from src.transcriber import ReplicateTranscriber
from src.speaker_labeler import SpeakerLabeler
from src.utils import setup_colab_environment
from src.data_models import Utterance

# Import configuration
try:
    from config import INPUT_DIR, OUTPUT_DIR, HF_TOKEN, REPLICATE_TOKEN
except ImportError:
    logging.error("Could not import configuration from config.py. Please ensure it exists and is in the same directory.")
    INPUT_DIR = "."
    OUTPUT_DIR = "./processed_output"
    HF_TOKEN = "dummy_hf_token"
    REPLICATE_TOKEN = "dummy_replicate_token"

# Configure logging (consistent with other modules)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PodcastPipeline:
    """Main pipeline orchestrator with audio downloading capability"""

    def __init__(self, input_dir: str, output_dir: str, hf_token: str, replicate_token: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)

        # Initialize output files
        self.episodes_file = self.output_dir / "episodes.jsonl"
        self.errors_file = self.output_dir / "errors.log"
        self.download_log_file = self.output_dir / "download.log"

        # Initialize components
        self.audio_downloader = AudioDownloader(str(self.input_dir))
        self.audio_processor = AudioProcessor()
        self.file_uploader = FileUploader()
        self.transcriber = ReplicateTranscriber(hf_token, replicate_token, blend_threshold=2.0)
        self.speaker_labeler = SpeakerLabeler()

        logger.info("PodcastPipeline initialized successfully")

    def download_playlist_audio(self, playlist_url: str) -> bool:
        """
        Download all audio files from a YouTube playlist
        
        Args:
            playlist_url: URL of the YouTube playlist to download
            
        Returns:
            True if download was successful, False otherwise
        """
        logger.info(f"Starting playlist download from: {playlist_url}")
        
        try:
            # Log download start
            self._log_download_info(f"Starting download from playlist: {playlist_url}")
            
            # Validate URL first
            if not self.audio_downloader.validate_url(playlist_url):
                error_msg = f"Invalid or inaccessible playlist URL: {playlist_url}"
                logger.error(error_msg)
                self._log_download_error(error_msg)
                return False
            
            # Download playlist
            downloaded_files = self.audio_downloader.download_playlist(playlist_url)
            
            if not downloaded_files:
                error_msg = f"No files were downloaded from playlist: {playlist_url}"
                logger.error(error_msg)
                self._log_download_error(error_msg)
                return False
            
            # Log successful downloads
            self._log_download_info(f"Successfully downloaded {len(downloaded_files)} files")
            for file_path in downloaded_files:
                file_name = Path(file_path).name
                self._log_download_info(f"Downloaded: {file_name}")
            
            # Clean up metadata files
            self.audio_downloader.cleanup_metadata_files()
            
            logger.info(f"✅ Playlist download completed successfully - {len(downloaded_files)} files downloaded")
            return True
            
        except Exception as e:
            error_msg = f"Playlist download failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_download_error(error_msg)
            return False

    def download_single_video(self, video_url: str) -> bool:
        """
        Download audio from a single YouTube video
        
        Args:
            video_url: URL of the YouTube video to download
            
        Returns:
            True if download was successful, False otherwise
        """
        logger.info(f"Starting single video download from: {video_url}")
        
        try:
            # Log download start
            self._log_download_info(f"Starting download from video: {video_url}")
            
            # Validate URL first
            if not self.audio_downloader.validate_url(video_url):
                error_msg = f"Invalid or inaccessible video URL: {video_url}"
                logger.error(error_msg)
                self._log_download_error(error_msg)
                return False
            
            # Download video
            downloaded_file = self.audio_downloader.download_single_video(video_url)
            
            if not downloaded_file:
                error_msg = f"Failed to download video: {video_url}"
                logger.error(error_msg)
                self._log_download_error(error_msg)
                return False
            
            # Log successful download
            file_name = Path(downloaded_file).name
            self._log_download_info(f"Successfully downloaded: {file_name}")
            
            # Clean up metadata files
            self.audio_downloader.cleanup_metadata_files()
            
            logger.info(f"✅ Single video download completed successfully: {file_name}")
            return True
            
        except Exception as e:
            error_msg = f"Single video download failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_download_error(error_msg)
            return False

    def process_episode(self, episode_path: Path) -> bool:
        """Process a single episode with comprehensive error handling"""
        episode_path = Path(episode_path)
        episode_id = episode_path.stem
        logger.info(f"Processing episode: {episode_id}")

        try:
            # Stage 1: Preprocess audio
            temp_audio_path = self.temp_dir / f"{episode_id}_clean.wav"

            logger.info(f"Stage 1: Preprocessing audio for {episode_id} from {episode_path}")
            if not self.audio_processor.preprocess_audio(str(episode_path), str(temp_audio_path)):
                self._log_error(episode_id, "Audio preprocessing failed")
                temp_audio_path.unlink(missing_ok=True)
                return False

            # Stage 2: Upload audio to get public link
            logger.info(f"Stage 2: Uploading audio for {episode_id} from {temp_audio_path}")
            audio_url = self.file_uploader.upload_file(str(temp_audio_path))
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
            logger.error(error_msg, exc_info=True)
            self._log_error(episode_id, error_msg)
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

    def _log_download_info(self, message: str):
        """Log download information to download log file"""
        try:
            from datetime import datetime
            with open(self.download_log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} - INFO: {message}\n")
        except Exception as e:
            logger.error(f"Failed to write download info to log file: {e}")

    def _log_download_error(self, error_message: str):
        """Log download error to download log file"""
        try:
            from datetime import datetime
            with open(self.download_log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} - ERROR: {error_message}\n")
        except Exception as e:
            logger.error(f"Failed to write download error to log file: {e}")

    def _save_episode_results(self, utterances: List[Utterance]):
        """Save episode results to episodes.jsonl file"""
        try:
            with open(self.episodes_file, 'a', encoding='utf-8') as f:
                for utterance in utterances:
                    f.write(json.dumps(asdict(utterance), ensure_ascii=False) + '\n')
            logger.debug(f"Saved {len(utterances)} utterances to {self.episodes_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {self.episodes_file}: {e}")

    def run_full_pipeline(self, playlist_url: str = None, skip_download: bool = False):
        """
        Run the complete pipeline including downloading and processing
        
        Args:
            playlist_url: URL of the YouTube playlist to download (optional)
            skip_download: If True, skip download and process existing files
        """
        logger.info("🚀 Starting Full Podcast Pipeline...")
        
        # Stage 0: Download audio (if not skipped)
        if not skip_download:
            if not playlist_url:
                # Use the default playlist URL from audio_downloader.py
                playlist_url = "https://www.youtube.com/watch?v=upgAxjEZ7YU&list=PLfeJT8wCesumA5kQvGS4SsFVopuuFAFQ8"
                logger.info(f"Using default playlist URL: {playlist_url}")
            
            logger.info(f"\n{'='*60}")
            logger.info("STAGE 0: DOWNLOADING AUDIO")
            logger.info(f"{'='*60}")
            
            if not self.download_playlist_audio(playlist_url):
                logger.error("❌ Download failed. Cannot proceed with processing.")
                return
            
            logger.info("✅ Download stage completed successfully")
        else:
            logger.info("⏭️ Skipping download stage as requested")
        
        # Continue with existing pipeline logic
        self.run_processing_pipeline()

    def run_processing_pipeline(self):
        """Run the processing pipeline on existing audio files"""
        logger.info(f"\n{'='*60}")
        logger.info("STAGE 1-5: PROCESSING AUDIO FILES")
        logger.info(f"{'='*60}")
        
        # Find all episode files
        episode_files = []
        supported_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac", "*.ogg"]
        for pattern in supported_extensions:
            episode_files.extend(self.input_dir.glob(pattern))

        episode_files.sort()

        if not episode_files:
            logger.error(f"No audio files found in input directory: {self.input_dir}")
            logger.info("Please ensure audio files are in the specified INPUT_DIR or run with download enabled")
            return

        logger.info(f"Found {len(episode_files)} episodes to process in {self.input_dir}")

        # Process each episode
        success_count = 0
        start_time = time.time()

        for i, episode_file in enumerate(episode_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing episode {i}/{len(episode_files)}: {episode_file.name}")
            logger.info(f"{'='*60}")

            if self.process_episode(episode_file):
                success_count += 1
            else:
                logger.error(f"❌ Failed to process: {episode_file.name}")

            # Log progress
            elapsed = time.time() - start_time
            avg_time = elapsed / i if i > 0 else 0
            remaining = (len(episode_files) - i) * avg_time if avg_time > 0 else 0

            logger.info(f"Progress: {i}/{len(episode_files)} | "
                       f"Success: {success_count} | "
                       f"Avg time: {avg_time/60:.1f}m/episode | "
                       f"ETA: {remaining/60:.1f}m")

            # Memory cleanup
            gc.collect()
            logger.debug("Ran garbage collection.")

        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Processing Pipeline completed!")
        logger.info(f"📊 Results: {success_count}/{len(episode_files)} episodes processed successfully")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Output file: {self.episodes_file}")
        logger.info(f"📁 Error log: {self.errors_file}")
        logger.info(f"📁 Download log: {self.download_log_file}")

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
                lines = []
                for _ in range(5):  # Read up to 5 lines
                    try:
                        line = next(f)
                        lines.append(line)
                    except StopIteration:
                        break

            if not lines:
                logger.info("Output file is empty.")
                return

            for line in lines:
                try:
                    utterance = json.loads(line)
                    episode_id = utterance.get('episode_id', 'N/A')
                    speaker = utterance.get('speaker', 'N/A')
                    start = utterance.get('start', 0.0)
                    end = utterance.get('end', 0.0)
                    text = utterance.get('text', '').strip()

                    logger.info(f"  Episode: {episode_id} | "
                               f"Speaker: {speaker:<5} | "
                               f"Time: {start:.1f}s-{end:.1f}s | "
                               f"Text: {text[:80]}...")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line as JSON: {line.strip()[:100]}...")
                except Exception as e:
                    logger.warning(f"Error processing line from output file: {e}")

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

    # Validate tokens
    if HF_TOKEN == "your_huggingface_token_here" or REPLICATE_TOKEN == "your_replicate_token_here":
        logger.error("❌ Please update the HF_TOKEN and REPLICATE_TOKEN variables in config.py with your actual tokens")
        logger.info("Get your tokens from:")
        logger.info("  - HuggingFace: https://huggingface.co/settings/tokens")
        logger.info("  - Replicate: https://replicate.com/account/api-tokens")
        return

    # Create and run pipeline
    try:
        logger.info("Starting Full Podcast Pipeline with Download...")
        pipeline = PodcastPipeline(INPUT_DIR, OUTPUT_DIR, HF_TOKEN, REPLICATE_TOKEN)
        
        # Run full pipeline (download + process)
        # You can customize this by:
        # 1. Providing a different playlist URL
        # 2. Setting skip_download=True to only process existing files
        pipeline.run_full_pipeline(
            playlist_url=None,  # Will use default from audio_downloader.py
            skip_download=False  # Set to True to skip download
        )
        
        logger.info("Full Podcast Pipeline finished.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()