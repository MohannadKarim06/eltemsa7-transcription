import os
import logging
import yt_dlp
import re
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Arabic to English digit map
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

class AudioDownloader:
    """Handles downloading audio and renaming files with consistent episode IDs"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writeinfojson': True,
            'ignoreerrors': True,
            'no_warnings': False,
            'extractaudio': True,
            'audioformat': 'mp3',
        }

        logger.info(f"AudioDownloader initialized with output directory: {self.output_dir}")

    def download_playlist(self, playlist_url: str) -> List[str]:
        downloaded_files = []

        try:
            logger.info(f"Downloading from playlist: {playlist_url}")

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([playlist_url])

            downloaded_files = self._find_downloaded_files()
            logger.info(f"Downloaded {len(downloaded_files)} files")

            renamed_files = self._rename_to_episode_ids(downloaded_files)
            logger.info(f"{len(renamed_files)} valid episode files after renaming")
            return renamed_files

        except Exception as e:
            logger.error(f"Error downloading playlist: {e}")
            return []

    def download_single_video(self, video_url: str) -> Optional[str]:
        try:
            logger.info(f"Downloading single video: {video_url}")
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([video_url])

            downloaded_files = self._find_downloaded_files()
            if not downloaded_files:
                return None

            renamed_files = self._rename_to_episode_ids(downloaded_files)
            return renamed_files[0] if renamed_files else None

        except Exception as e:
            logger.error(f"Error downloading single video: {e}")
            return None

    def _find_downloaded_files(self) -> List[str]:
        return [str(f) for f in self.output_dir.glob("*.mp3")]

    def _rename_to_episode_ids(self, file_paths: List[str]) -> List[str]:
        renamed_files = []
        used_numbers = set()

        for path in file_paths:
            path = Path(path)
            episode_number = self._extract_arabic_episode_number(path.stem)

            if episode_number is None:
                logger.warning(f"No episode number found in title: {path.name}. Deleting.")
                path.unlink(missing_ok=True)
                continue

            # Avoid duplicates
            if episode_number in used_numbers:
                logger.warning(f"Duplicate episode number {episode_number} detected. Skipping file: {path.name}")
                path.unlink(missing_ok=True)
                continue

            used_numbers.add(episode_number)
            new_name = f"episode_{int(episode_number):03d}{path.suffix}"
            new_path = path.with_name(new_name)

            try:
                path.rename(new_path)
                renamed_files.append(str(new_path))
                logger.info(f"Renamed {path.name} → {new_name}")
            except Exception as e:
                logger.error(f"Failed to rename {path.name}: {e}")
                continue

        return renamed_files

    def _extract_arabic_episode_number(self, text: str) -> Optional[int]:
        # Arabic digits: ٠١٢٣٤٥٦٧٨٩
        matches = re.findall(r'[٠١٢٣٤٥٦٧٨٩]+', text)
        if matches:
            try:
                number = int(matches[0].translate(ARABIC_DIGITS))
                return number
            except ValueError:
                return None
        return None

    def cleanup_metadata_files(self):
        for f in self.output_dir.glob("*.info.json"):
            f.unlink(missing_ok=True)

    def get_existing_files(self) -> List[str]:
        return self._find_downloaded_files()

    def validate_url(self, url: str) -> bool:
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info is not None
        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            return False
