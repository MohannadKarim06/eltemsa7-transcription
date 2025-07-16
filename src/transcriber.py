import json
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import replicate

# Configure logging (consistent with other modules)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Utterance:
    """Individual utterance with speaker and timing"""
    episode_id: str
    start: float
    end: float
    speaker: str
    text: str


class EnhancedTranscriptionProcessor:
    """Enhanced processor for word-level transcription with utterance blending"""

    def __init__(self, blend_threshold: float = 2.0):
        """
        Initialize processor

        Args:
            blend_threshold: Maximum time gap (seconds) between utterances to blend them
        """
        self.blend_threshold = blend_threshold
        logger.info(f"EnhancedTranscriptionProcessor initialized with blend_threshold: {self.blend_threshold}")

    def process_transcription_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process transcription result with word-level extraction and utterance blending

        Args:
            result: Raw transcription result from Replicate

        Returns:
            Dict with processed segments ready for speaker labeling
        """
        if "output" not in result:
            logger.error(f"No 'output' field in response: {list(result.keys())}")
            return {"segments": []}

        output = result["output"]

        if output is None:
            logger.error("Replicate returned None output")
            return {"segments": []}

        # Extract segments from different possible formats
        segments = []
        if isinstance(output, dict):
            if "segments" in output:
                segments = output["segments"]
            else:
                logger.error(f"No 'segments' key in output. Keys: {list(output.keys())}")
                return {"segments": []}
        elif isinstance(output, list):
            segments = output
        else:
            logger.error(f"Unexpected output format: {type(output)}")
            return {"segments": []}


        if not segments:
            logger.warning("No segments found in transcription result")
            return {"segments": []}

        logger.info(f"Found {len(segments)} raw segments")

        # Step 1: Extract word-level utterances
        word_utterances = self._extract_word_utterances(segments)
        logger.info(f"Extracted {len(word_utterances)} word-level utterances")

        # Step 2: Blend close utterances by same speaker
        blended_utterances = self._blend_utterances(word_utterances)
        logger.info(f"Blended into {len(blended_utterances)} utterances")

        # Step 3: Validate final utterances
        valid_utterances = [u for u in blended_utterances if self._validate_utterance(u)]
        logger.info(f"Valid utterances: {len(valid_utterances)}")


        return {
            "segments": valid_utterances,
            "detected_language": output.get("detected_language", "ar") if isinstance(output, dict) else "ar" # Handle potential list output
        }

    def _extract_word_utterances(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract word-level utterances from segments

        Args:
            segments: List of segment dictionaries

        Returns:
            List of word-level utterances
        """
        word_utterances = []

        for segment in segments:
            if not self._validate_segment(segment):
                continue

            # Get words from segment
            words = segment.get("words", [])

            if not words:
                # If no words, create utterance from segment level
                # Ensure start/end exist before creating
                start = segment.get("start")
                end = segment.get("end")
                text = segment.get("text", "").strip()

                if start is not None and end is not None and text:
                    word_utterances.append({
                        "start": start,
                        "end": end,
                        "text": text,
                        "speaker": segment.get("speaker", "UNKNOWN")
                    })
                continue

            # Create utterance for each word
            for word_info in words:
                if not self._validate_word(word_info):
                    continue

                utterance = {
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                    "text": word_info.get("word", "").strip(),
                    "speaker": word_info.get("speaker", segment.get("speaker", "UNKNOWN"))
                }

                if utterance["text"]:  # Only add non-empty text
                    word_utterances.append(utterance)

        # Sort by start time
        word_utterances.sort(key=lambda x: x["start"])

        return word_utterances

    def _blend_utterances(self, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Blend utterances that are close in time and from the same speaker

        Args:
            utterances: List of word-level utterances

        Returns:
            List of blended utterances
        """
        if not utterances:
            return []

        blended = []
        current_utterance = utterances[0].copy()

        for i in range(1, len(utterances)):
            next_utterance = utterances[i]

            # Check if we should blend with current utterance
            should_blend = (
                next_utterance.get("speaker") == current_utterance.get("speaker") and
                (next_utterance.get("start", 0) - current_utterance.get("end", 0)) <= self.blend_threshold and
                next_utterance.get("start", 0) >= current_utterance.get("start", 0) # Ensure chronological order
            )

            if should_blend:
                # Blend utterances
                current_utterance["end"] = next_utterance.get("end", current_utterance["end"])
                current_utterance["text"] = (
                    current_utterance.get("text", "") + " " + next_utterance.get("text", "")
                ).strip()
            else:
                # Save current utterance and start new one
                blended.append(current_utterance)
                current_utterance = next_utterance.copy()

        # Don't forget the last utterance
        blended.append(current_utterance)

        return blended

    def _validate_segment(self, segment: Dict[str, Any]) -> bool:
        """Validate a transcription segment"""
        if not isinstance(segment, dict):
            logger.debug(f"Invalid segment format: {segment}")
            return False

        # Check if has either text or words
        has_text = segment.get("text", "").strip()
        has_words = segment.get("words", [])

        if not has_text and not has_words:
            logger.debug(f"Segment has no text or words: {segment}")
            return False

        # Check timing if available
        start = segment.get("start", 0)
        end = segment.get("end", 0)

        if start is None or end is None or start < 0 or end < 0 or (end > 0 and start >= end):
             logger.debug(f"Invalid segment timing: start={start}, end={end} ({segment})")
             return False


        return True

    def _validate_word(self, word_info: Dict[str, Any]) -> bool:
        """Validate a word-level entry"""
        if not isinstance(word_info, dict):
            logger.debug(f"Invalid word format: {word_info}")
            return False

        # Check if has word text
        if not word_info.get("word", "").strip():
            logger.debug(f"Word entry has no text: {word_info}")
            return False

        # Check timing
        start = word_info.get("start", 0)
        end = word_info.get("end", 0)

        if start is None or end is None or start < 0 or end < 0 or start >= end:
            logger.debug(f"Invalid word timing: start={start}, end={end} ({word_info})")
            return False

        return True

    def _validate_utterance(self, utterance: Dict[str, Any]) -> bool:
        """Validate a final utterance"""
        if not isinstance(utterance, dict):
            logger.debug(f"Invalid utterance format: {utterance}")
            return False

        # Check required fields
        required_fields = ["start", "end", "text", "speaker"]
        if not all(field in utterance for field in required_fields):
            logger.debug(f"Utterance missing required fields: {utterance}")
            return False

        # Check text is not empty
        if not utterance["text"].strip():
            logger.debug(f"Utterance has empty text: {utterance}")
            return False

        # Check timing
        start = utterance["start"]
        end = utterance["end"]

        if start is None or end is None or start < 0 or end < 0 or start >= end:
            logger.debug(f"Invalid utterance timing: start={start}, end={end} ({utterance})")
            return False

        return True


class ReplicateTranscriber:
    """Handles transcription using Replicate WhisperX with enhanced processing"""

    def __init__(self, hf_token: str, replicate_token: str, blend_threshold: float = 2.0):
        self.hf_token = hf_token
        self.replicate_token = replicate_token
        # Using a specific version for stability
        self.model_version = "84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
        self.base_url = "https://api.replicate.com/v1/predictions"

        # Initialize enhanced processor
        self.processor = EnhancedTranscriptionProcessor(blend_threshold)
        logger.info("ReplicateTranscriber initialized")


    def transcribe_with_diarization(self, audio_url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Transcribe audio using Replicate WhisperX with enhanced processing"""

        # Validate audio URL
        if not self._validate_audio_url(audio_url):
            logger.error(f"Audio URL validation failed: {audio_url}")
            return {"segments": []}

        for attempt in range(max_retries):
            try:
                logger.info(f"Starting transcription attempt {attempt + 1} for: {audio_url}")

                # Prepare the request payload
                payload = {
                    "version": self.model_version,
                    "input": {
                        "audio_file": audio_url,
                        "language": "ar", # Specify language
                        "diarization": True,
                        "align_output": True,
                        "huggingface_access_token": self.hf_token,
                        "batch_size": 32, # Recommended batch size
                        "temperature": 0.0,
                        "debug": False
                    }
                }

                # Use sync mode for better error handling
                headers = {
                    "Authorization": f"Bearer {self.replicate_token}",
                    "Content-Type": "application/json",
                    "Prefer": "wait" # Use wait header for sync processing
                }

                logger.info(f"Making HTTP POST request to Replicate API for {audio_url}...")

                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=1200  # 20 minutes timeout for the initial request
                )

                # Handle response
                if response.status_code == 200:
                    logger.info("Initial sync request succeeded (status 200)")
                    try:
                        result = response.json()
                        # Use enhanced processor
                        return self.processor.process_transcription_result(result)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response from 200 status: {e}")
                        logger.error(f"Response text: {response.text}")
                        return {"segments": []} # Return empty on JSON error

                elif response.status_code in [201, 202]:
                    # Handle async response (prediction created)
                    logger.info(f"Prediction created (status {response.status_code}), polling...")
                    result = response.json()
                    prediction_id = result.get("id")
                    if prediction_id:
                        return self._poll_prediction_completion(prediction_id)
                    else:
                         logger.error("Received 201/202 but no prediction ID in response.")
                         return {"segments": []} # Return empty if no ID

                else:
                    logger.error(f"HTTP error {response.status_code} during transcription request for {audio_url}: {response.text}")
                    # Do not retry on likely permanent errors like 401, 403, 404, 422
                    if response.status_code in [401, 403, 404, 422]:
                         logger.error(f"Stopping retries for {audio_url} due to status code {response.status_code}")
                         return {"segments": []}


            except requests.exceptions.Timeout:
                logger.error(f"Request timeout on attempt {attempt + 1} for {audio_url}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1} for {audio_url}: {e}")

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1} for {audio_url}: {e}")

            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 15 # Exponential backoff: 15, 30, 60 seconds
                logger.info(f"Waiting {wait_time}s before retry for {audio_url}...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) exhausted for {audio_url}.")


        logger.error(f"All transcription attempts failed for {audio_url}")
        return {"segments": []} # Return empty if all attempts fail


    def _poll_prediction_completion(self, prediction_id: str, max_wait_time: int = 1800) -> Dict[str, Any]:
        """Poll for prediction completion with enhanced processing"""
        logger.info(f"Polling prediction {prediction_id} for completion...")

        start_time = time.time()
        poll_interval = 15
        consecutive_errors = 0
        consecutive_timeouts = 0
        max_consecutive_errors = 5
        max_consecutive_timeouts = 3


        while time.time() - start_time < max_wait_time:
            try:
                status_response = self._check_prediction_status(prediction_id)

                if not status_response:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors ({max_consecutive_errors}) while polling prediction {prediction_id}")
                        break
                    logger.warning(f"Received empty status response for prediction {prediction_id}. Consecutive errors: {consecutive_errors}")
                    time.sleep(poll_interval)
                    continue

                consecutive_errors = 0 # Reset error counter on success
                consecutive_timeouts = 0 # Reset timeout counter on success

                status = status_response.get("status", "unknown")
                elapsed = time.time() - start_time

                logger.info(f"Prediction {prediction_id} status: {status} (elapsed: {elapsed:.1f}s)")

                if status == "succeeded":
                    logger.info(f"Prediction {prediction_id} succeeded.")
                    # Use enhanced processor
                    return self.processor.process_transcription_result(status_response)

                elif status == "failed":
                    error = status_response.get("error", "Unknown error")
                    logger.error(f"Prediction {prediction_id} failed: {error}")
                    return {"segments": []}

                elif status in ["starting", "processing"]:
                    # Adaptive polling
                    if elapsed > 1200: # After 20 minutes
                        poll_interval = 120 # Poll every 2 minutes
                    elif elapsed > 600:  # After 10 minutes
                        poll_interval = 60   # Poll every 1 minute
                    elif elapsed > 300:  # After 5 minutes
                        poll_interval = 30   # Poll every 30 seconds

                    time.sleep(poll_interval)

                elif status == "canceled":
                    logger.warning(f"Prediction {prediction_id} was canceled.")
                    return {"segments": []}

                else:
                    logger.warning(f"Unknown status '{status}' for prediction {prediction_id}. Waiting...")
                    time.sleep(poll_interval)

            except requests.exceptions.Timeout:
                 consecutive_timeouts += 1
                 if consecutive_timeouts >= max_consecutive_timeouts:
                      logger.error(f"Too many consecutive timeouts ({max_consecutive_timeouts}) while polling prediction {prediction_id}")
                      break
                 logger.warning(f"Polling request timed out for prediction {prediction_id}. Consecutive timeouts: {consecutive_timeouts}")
                 time.sleep(poll_interval) # Wait before next attempt

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error while polling prediction {prediction_id}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({max_consecutive_errors}) while polling prediction {prediction_id}")
                    break
                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Unexpected error while polling prediction {prediction_id}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({max_consecutive_errors}) while polling prediction {prediction_id}")
                    break
                time.sleep(poll_interval)


        logger.error(f"Polling for prediction {prediction_id} timed out after {max_wait_time} seconds")
        return {"segments": []} # Return empty if polling times out


    def _validate_audio_url(self, url: str) -> bool:
        """Validate that the audio URL is accessible"""
        try:
            if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                logger.error(f"Invalid or empty URL format provided: {url}")
                return False

            # Try HEAD request first
            try:
                response = requests.head(url, timeout=60, allow_redirects=True)
                if response.status_code == 200:
                    logger.debug(f"HEAD request successful for {url}")
                    return self._check_url_response(response, url)
            except Exception as e:
                logger.debug(f"HEAD request failed for {url}: {e}")
                pass # Ignore and try GET


            # If HEAD fails or is not supported, try GET with range
            try:
                logger.debug(f"Attempting GET request with range for {url}")
                response = requests.get(url, headers={'Range': 'bytes=0-1024'}, timeout=60, allow_redirects=True)
                if response.status_code in [200, 206]:
                    logger.debug(f"GET request with range successful for {url}")
                    return self._check_url_response(response, url)
            except Exception as e:
                logger.debug(f"GET request with range failed for {url}: {e}")
                pass # Ignore and fail validation


            logger.error(f"URL not accessible after HEAD and GET attempts: {url}")
            return False

        except Exception as e:
            logger.error(f"URL validation error for {url}: {e}")
            return False


    def _check_url_response(self, response: requests.Response, url: str) -> bool:
        """Helper method to check URL response for content type and length"""
        content_type = response.headers.get('content-type', '')
        if content_type:
            logger.debug(f"Content-Type for {url}: {content_type}")
            if not any(t in content_type.lower() for t in ['audio', 'wav', 'mp3', 'flac', 'm4a', 'octet-stream']):
                logger.warning(f"Unexpected content type '{content_type}' for {url}. Proceeding but be aware.")
        else:
             logger.warning(f"No Content-Type header for {url}. Cannot verify file type.")


        content_length = response.headers.get('content-length')
        if content_length:
            try:
                size_bytes = int(content_length)
                size_mb = size_bytes / (1024 * 1024)
                logger.info(f"Audio file size: {size_mb:.2f} MB for {url}")

                # Limit file size to prevent excessive costs or processing time
                max_file_size_mb = 500 # Define a reasonable limit
                if size_mb > max_file_size_mb:
                    logger.error(f"Audio file too large ({size_mb:.2f} MB) for {url}. Max allowed: {max_file_size_mb} MB.")
                    return False
            except ValueError:
                logger.warning(f"Could not parse Content-Length: {content_length} for {url}")
            except Exception as e:
                 logger.warning(f"Error processing Content-Length for {url}: {e}")

        else:
             logger.warning(f"No Content-Length header for {url}. Cannot verify file size.")


        return True


    def _check_prediction_status(self, prediction_id: str) -> Dict[str, Any]:
        """Check prediction status with better error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {self.replicate_token}",
                "Content-Type": "application/json"
            }

            response = requests.get(
                f"{self.base_url}/{prediction_id}",
                headers=headers,
                timeout=30 # Short timeout for polling check
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to check prediction status {prediction_id}: HTTP {response.status_code} - {response.text}")
                return {} # Return empty on error


        except requests.exceptions.Timeout:
             logger.warning(f"Timeout while checking status for prediction {prediction_id}")
             return {} # Return empty on timeout
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while checking status for prediction {prediction_id}: {e}")
            return {} # Return empty on request error
        except Exception as e:
            logger.error(f"Unexpected error checking status for prediction {prediction_id}: {e}")
            return {} # Return empty on unexpected error