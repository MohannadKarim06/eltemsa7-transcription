import os
import json
import subprocess
import logging
import time
import requests
from pathlib import Path
from typing import Optional

# Configure logging (consistent with other modules)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileUploader:
    """Handles uploading files to get public direct links"""

    def __init__(self):
      self.services = [
          self._upload_to_transfersh, # Try this first - more reliable
          self._upload_to_pomf,       # Fixed URL
          self._upload_to_uguu,       # Usually works
          self._upload_to_catbox,     # Has accessibility issues
          self._upload_to_tmpfiles,   # Fallback
          self._upload_to_0x0st,      # Last resort due to 403 errors
      ]
      logger.info("FileUploader initialized")


    def upload_file(self, file_path: str, max_retries: int = 3) -> Optional[str]:
        """Upload file to a service and return public direct link"""
        # Verify file exists and is not empty
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"File is empty: {file_path}")
            return None

        logger.info(f"Uploading file: {file_path} ({file_size} bytes)")

        for service in self.services:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to upload using {service.__name__} (attempt {attempt + 1})")
                    url = service(file_path)
                    if url:
                        # Test if URL is accessible
                        if self._test_url_accessibility(url):
                            logger.info(f"Successfully uploaded to: {url}")
                            return url
                        else:
                            logger.warning(f"URL not accessible: {url}")

                except Exception as e:
                    logger.warning(f"Upload attempt {attempt + 1} failed with {service.__name__}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Fixed delay instead of exponential
                        logger.error(f"Failed to upload {file_path} after trying all services")
                        return None
            logger.error(f"Failed to upload {file_path} after trying all services")
            return None # Ensure None is returned if all services fail after retries


    def _test_url_accessibility(self, url: str) -> bool:
        """Test if URL is accessible with HEAD request"""
        try:
            response = requests.head(url, timeout=30, allow_redirects=True)
            if response.status_code == 200:
                return True
            # If HEAD fails, try GET request with range header
            response = requests.get(url, headers={'Range': 'bytes=0-1024'}, timeout=30)
            return response.status_code in [200, 206]
        except Exception as e: # Catch specific exceptions for clarity
            logger.warning(f"URL accessibility test failed for {url}: {e}")
            return False

    def _upload_to_catbox(self, file_path: str) -> Optional[str]:
        """Upload to catbox.moe (permanent, good for large files)"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://catbox.moe/user/api.php',
                    data={'reqtype': 'fileupload'},
                    files={'fileToUpload': f},
                    timeout=300
                )

            if response.status_code == 200:
                url = response.text.strip()
                if url.startswith('https://files.catbox.moe/'):
                    # Test the URL immediately after upload
                    test_response = requests.head(url, timeout=30)
                    if test_response.status_code == 200:
                        return url
                    else:
                        logger.warning(f"Catbox URL not immediately accessible: {url}")
                        # Wait a bit and try again
                        time.sleep(5)
                        test_response = requests.head(url, timeout=30)
                        if test_response.status_code == 200:
                            return url

            logger.error(f"catbox.moe upload failed: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"catbox.moe upload error: {e}")
            return None

    def _upload_to_0x0st(self, file_path: str) -> Optional[str]:
        """Upload to 0x0.st (365 days expiry)"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://0x0.st',
                    files={'file': f},
                    timeout=300
                )

            if response.status_code == 200:
                url = response.text.strip()
                if url.startswith('https://'):
                    return url

            logger.error(f"0x0.st upload failed: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"0x0.st upload error: {e}")
            return None

    def _upload_to_pomf(self, file_path: str) -> Optional[str]:
        """Upload to pomf.lain.la (72 hours expiry)"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://pomf.lain.la/upload.php',
                    files={'files[]': f},
                    timeout=600 # Increased timeout
                )

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data['files'][0]['url']

            logger.error(f"pomf.lain.la upload failed: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"pomf.lain.la upload error: {e}")
            return None

    def _upload_to_uguu(self, file_path: str) -> Optional[str]:
        """Upload to uguu.se (24 hours expiry)"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://uguu.se/upload',
                    files={'files[]': f},
                    timeout=300
                )

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data['files'][0]['url']

            logger.error(f"uguu.se upload failed: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"uguu.se upload error: {e}")
            return None

    def _upload_to_transfersh(self, file_path: str) -> Optional[str]:
        """Upload to transfer.sh (14 days expiry)"""
        try:
            filename = Path(file_path).name
            with open(file_path, 'rb') as f:
                response = requests.put(
                    f'https://transfer.sh/{filename}',
                    data=f,
                    timeout=300
                )

            if response.status_code == 200:
                return response.text.strip()

            logger.error(f"transfer.sh upload failed: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"transfer.sh upload error: {e}")
            return None

    def _upload_to_tmpfiles(self, file_path: str) -> Optional[str]:
        """Upload to tmpfiles.org (1 hour expiry)"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://tmpfiles.org/api/v1/upload',
                    files={'file': f},
                    timeout=300
                )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    url = data['data']['url']
                    # Convert to direct link
                    direct_url = url.replace('tmpfiles.org/dl/', 'tmpfiles.org/')
                    return direct_url

            logger.error(f"tmpfiles.org upload failed: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"tmpfiles.org upload error: {e}")
            return None