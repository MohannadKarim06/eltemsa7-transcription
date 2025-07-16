import os
import logging
import yt_dlp
from pathlib import Path
from typing import List, Optional

# Configure logging (consistent with other modules)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioDownloader:
    """Handles downloading audio from YouTube playlists or individual videos"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writeinfojson': True,  # Save metadata
            'writethumbnail': False,
            'ignoreerrors': True,  # Continue on errors
            'no_warnings': False,
            'extractaudio': True,
            'audioformat': 'mp3',
            'embed_subs': False,
            'writesubtitles': False,
        }
        
        logger.info(f"AudioDownloader initialized with output directory: {self.output_dir}")

    def download_playlist(self, playlist_url: str) -> List[str]:
        """
        Download all audio files from a YouTube playlist
        
        Args:
            playlist_url: URL of the YouTube playlist
            
        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        
        try:
            logger.info(f"Starting playlist download from: {playlist_url}")
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # First, extract playlist info to get video count
                try:
                    playlist_info = ydl.extract_info(playlist_url, download=False)
                    video_count = len(playlist_info.get('entries', []))
                    logger.info(f"Found {video_count} videos in playlist")
                except Exception as e:
                    logger.error(f"Failed to extract playlist info: {e}")
                    return []
                
                # Download the playlist
                try:
                    ydl.download([playlist_url])
                    logger.info("Download process completed")
                except Exception as e:
                    logger.error(f"Download process failed: {e}")
                    return []
            
            # Find downloaded files
            downloaded_files = self._find_downloaded_files()
            logger.info(f"Successfully downloaded {len(downloaded_files)} audio files")
            
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Playlist download failed: {e}")
            return []

    def download_single_video(self, video_url: str) -> Optional[str]:
        """
        Download audio from a single YouTube video
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            logger.info(f"Downloading single video: {video_url}")
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find the downloaded file
            downloaded_files = self._find_downloaded_files()
            if downloaded_files:
                latest_file = max(downloaded_files, key=lambda f: Path(f).stat().st_mtime)
                logger.info(f"Successfully downloaded: {latest_file}")
                return latest_file
            else:
                logger.error("No files found after download")
                return None
                
        except Exception as e:
            logger.error(f"Single video download failed: {e}")
            return None

    def _find_downloaded_files(self) -> List[str]:
        """Find all downloaded audio files in the output directory"""
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']
        downloaded_files = []
        
        for ext in audio_extensions:
            pattern = f"*{ext}"
            files = list(self.output_dir.glob(pattern))
            downloaded_files.extend([str(f) for f in files])
        
        return sorted(downloaded_files)

    def cleanup_metadata_files(self):
        """Clean up metadata files (.info.json) created during download"""
        try:
            info_files = list(self.output_dir.glob("*.info.json"))
            for info_file in info_files:
                info_file.unlink()
                logger.debug(f"Cleaned up metadata file: {info_file}")
            
            if info_files:
                logger.info(f"Cleaned up {len(info_files)} metadata files")
                
        except Exception as e:
            logger.error(f"Failed to cleanup metadata files: {e}")

    def get_existing_files(self) -> List[str]:
        """Get list of existing audio files in the output directory"""
        return self._find_downloaded_files()

    def validate_url(self, url: str) -> bool:
        """Validate if the URL is a valid YouTube URL"""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info is not None
        except Exception as e:
            logger.error(f"URL validation failed for {url}: {e}")
            return False