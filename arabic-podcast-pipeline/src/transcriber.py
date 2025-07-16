import logging
import time
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import replicate

# Import Utterance dataclass (assuming it's in data_models.py)
# from .data_models import Utterance # This relative import will be used in the final structure

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedTranscriptionProcessor:
    """Enhanced processor for word-level transcription with utterance blending"""

    def __init__(self, blend_threshold: float = 2.0):
        """
        Initialize processor

        Args:
            blend_threshold: Maximum time gap (seconds) between utterances to blend them
        """
        self.blend_threshold = blend_threshold

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

        logger.info(f"Found {{len(segments)}} raw segments")

        # Step 1: Extract word-level utterances
        word_utterances = self._extract_word_utterances(segments)
        logger.info(f"Extracted {{len(word_utterances)}} word-level utterances")

        # Step 2: Blend close utterances by same speaker
        blended_utterances = self._blend_utterances(word_utterances)
        logger.info(f"Blended into {{len(blended_utterances)}} utterances")

        # Step 3: Validate final utterances
        valid_utterances = [u for u in blended_utterances if self._validate_utterance(u)]
        logger.info(f"Valid utterances: {{len(valid_utterances)}}")

        return {
            "segments": valid_utterances,
            "detected_language": output.get("detected_language", "ar") if isinstance(output, dict) else "ar"
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
                word_utterances.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
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
                next_utterance["speaker"] == current_utterance["speaker"] and
                (next_utterance["start"] - current_utterance["end"]) <= self.blend_threshold and
                next_utterance["start"] >= current_utterance["start"]  # Ensure chronological order
            )

            if should_blend:
                # Blend utterances
                current_utterance["end"] = next_utterance["end"]
                current_utterance["text"] = (
                    current_utterance["text"] + " " + next_utterance["text"]
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
            return False

        # Check if has either text or words
        has_text = segment.get("text", "").strip()
        has_words = segment.get("words", [])

        if not has_text and not has_words:
            return False

        # Check timing if available
        start = segment.get("start", 0)
        end = segment.get("end", 0)

        if start < 0 or end < 0 or (end > 0 and start >= end):
            return False

        return True

    def _validate_word(self, word_info: Dict[str, Any]) -> bool:
        """Validate a word-level entry"""
        if not isinstance(word_info, dict):
            return False

        # Check if has word text
        if not word_info.get("word", "").strip():
            return False

        # Check timing
        start = word_info.get("start", 0)
        end = word_info.get("end", 0)

        if start < 0 or end < 0 or start >= end:
            return False

        return True

    def _validate_utterance(self, utterance: Dict[str, Any]) -> bool:
        """Validate a final utterance"""
        if not isinstance(utterance, dict):
            return False

        # Check required fields
        required_fields = ["start", "end", "text", "speaker"]
        if not all(field in utterance for field in required_fields):
            return False

        # Check text is not empty
        if not utterance["text"].strip():
            return False

        # Check timing
        start = utterance["start"]
        end = utterance["end"]

        if start < 0 or end < 0 or start >= end:
            return False

        return True


class ReplicateTranscriber:
    """Handles transcription using Replicate WhisperX with enhanced processing"""

    def __init__(self, hf_token: str, replicate_token: str, model_version: str, base_url: str, blend_threshold: float = 2.0):
        self.hf_token = hf_token
        self.replicate_token = replicate_token
        self.model_version = model_version
        self.base_url = base_url

        # Initialize enhanced processor
        self.processor = EnhancedTranscriptionProcessor(blend_threshold)

    def transcribe_with_diarization(self, audio_url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Transcribe audio using Replicate WhisperX with enhanced processing"""

        # Validate audio URL
        if not self._validate_audio_url(audio_url):
            logger.error(f"Audio URL validation failed: {{audio_url}}")
            return {{"segments": []}}

        for attempt in range(max_retries):
            try:
                logger.info(f"Starting transcription attempt {{attempt + 1}} for: {{audio_url}}")

                # Prepare the request payload
                payload = {
                    "version": self.model_version,
                    "input": {
                        "audio_file": audio_url,
                        "language": "ar",
                        "diarization": True,
                        "align_output": True,
                        "huggingface_access_token": self.hf_token,
                        "batch_size": 32,
                        "temperature": 0.0,
                        "debug": False
                    }
                }

                # Use sync mode for better error handling
                headers = {
                    "Authorization": f"Bearer {{self.replicate_token}}",
                    "Content-Type": "application/json",
                    "Prefer": "wait"
                }

                logger.info(f"Making HTTP request to Replicate API...")

                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=1200  # 20 minutes timeout
                )

                # Handle response
                if response.status_code == 200:
                    try:
                        result = response.json()
                        # Use enhanced processor instead of original method
                        return self.processor.process_transcription_result(result)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {{e}}")
                        logger.error(f"Response text: {{response.text}}")

                elif response.status_code in [201, 202]:
                    # Handle async response
                    result = response.json()
                    prediction_id = result.get("id")
                    if prediction_id:
                        return self._poll_prediction_completion(prediction_id)

                else:
                    logger.error(f"HTTP error {{response.status_code}}: {{response.text}}")

            except requests.exceptions.Timeout:
                logger.error(f"Request timeout on attempt {{attempt + 1}}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {{attempt + 1}}: {{e}}")

            except Exception as e:
                logger.error(f"Unexpected error on attempt {{attempt + 1}}: {{e}}")

            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 15  # 15, 30, 60 seconds
                logger.info(f"Waiting {{wait_time}}s before retry...")
                time.sleep(wait_time)

        logger.error("All transcription attempts failed")
        return {{"segments": []}}

    def _poll_prediction_completion(self, prediction_id: str, max_wait_time: int = 1800) -> Dict[str, Any]:
        """Poll for prediction completion with enhanced processing"""
        logger.info(f"Polling prediction {{prediction_id}} for completion...")

        start_time = time.time()
        poll_interval = 15
        consecutive_errors = 0

        while time.time() - start_time < max_wait_time:
            try:
                status_response = self._check_prediction_status(prediction_id)

                if not status_response:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logger.error("Too many consecutive errors while polling")
                        break
                    time.sleep(poll_interval)
                    continue

                consecutive_errors = 0
                status = status_response.get("status", "unknown")
                elapsed = time.time() - start_time

                logger.info(f"Prediction status: {{status}} (elapsed: {{elapsed:.1f}}s)")

                if status == "succeeded":
                    # Use enhanced processor
                    return self.processor.process_transcription_result(status_response)

                elif status == "failed":
                    error = status_response.get("error", "Unknown error")
                    logger.error(f"Prediction failed: {{error}}")
                    return {{"segments": []}}

                elif status in ["starting", "processing"]:
                    # Adaptive polling
                    if elapsed > 600:  # After 10 minutes
                        poll_interval = 60
                    elif elapsed > 300:  # After 5 minutes
                        poll_interval = 30

                    time.sleep(poll_interval)

                else:
                    logger.warning(f"Unknown status: {{status}}")
                    time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error while polling prediction: {{e}}")
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    logger.error("Too many consecutive errors while polling")
                    break
                time.sleep(poll_interval)

        logger.error(f"Transcription timed out after {{max_wait_time}} seconds")
        return {{"segments": []}}

    # Keep all the existing validation and polling methods...
    def _validate_audio_url(self, url: str) -> bool:
        """Validate that the audio URL is accessible"""
        try:
            if not url.startswith(('http://', 'https://')):
                logger.error(f"Invalid URL format: {{url}}")
                return False

            # Try HEAD request first
            try:
                response = requests.head(url, timeout=60, allow_redirects=True)
                if response.status_code == 200:
                    return self._check_url_response(response, url)
            except:
                pass

            # If HEAD fails, try GET with range
            try:
                response = requests.get(url, headers={{'Range': 'bytes=0-1024'}}, timeout=60, allow_redirects=True)
                if response.status_code in [200, 206]:
                    return self._check_url_response(response, url)
            except:
                pass

            logger.error(f"URL not accessible: {{url}}")
            return False

        except Exception as e:
            logger.error(f"URL validation error: {{e}}")
            return False

    def _check_url_response(self, response, url: str) -> bool:
        """Helper method to check URL response"""
        content_type = response.headers.get('content-type', '')
        if content_type and not any(t in content_type.lower() for t in ['audio', 'wav', 'mp3', 'flac', 'm4a', 'octet-stream']):
            logger.warning(f"Unexpected content type: {{content_type}}")

        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            logger.info(f"Audio file size: {{size_mb:.2f}} MB")

            if size_mb > 500:
                logger.error(f"Audio file too large: {{size_mb:.2f}} MB")
                return False

        return True

    def _check_prediction_status(self, prediction_id: str) -> Dict[str, Any]:
        """Check prediction status with better error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {{self.replicate_token}}",
                "Content-Type": "application/json"
            }

            response = requests.get(
                f"{{self.base_url}}/{{prediction_id}}",
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to check prediction status: {{response.status_code}}")
                return {}

        except Exception as e:
            logger.error(f"Error checking prediction status: {{e}}")
            return {}