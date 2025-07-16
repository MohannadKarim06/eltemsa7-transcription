import logging
from typing import List, Dict, Any
from dataclasses import asdict # Needed if we use asdict inside this class

# Import the Utterance dataclass
from .data_models import Utterance # Assuming data_models is in the same package

# Configure logging (consistent with other modules)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpeakerLabeler:
    """Labels speakers as Host/Guest based on speaking time"""

    def __init__(self):
        logger.info("SpeakerLabeler initialized")


    def label_speakers(self, segments: List[Dict[str, Any]], episode_id: str) -> List[Utterance]:
        """Label speakers as Host/Guest based on total speaking time"""
        if not segments:
            logger.warning(f"No segments provided for episode {episode_id} to label speakers.")
            return []

        # Calculate total speaking time per speaker
        speaker_times = {}
        for segment in segments:
            # Validate segment structure before accessing keys
            if not isinstance(segment, dict):
                 logger.warning(f"Skipping invalid segment format for episode {episode_id}: {segment}")
                 continue

            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start")
            end = segment.get("end")

            # Validate timing
            if start is None or end is None or start < 0 or end < 0 or start >= end:
                 logger.warning(f"Skipping segment with invalid timing for episode {episode_id}: start={start}, end={end}")
                 continue

            duration = end - start

            if duration > 0:  # Only count valid durations
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

        # Determine host as speaker with most speaking time
        if not speaker_times:
            logger.warning(f"No speaker times calculated for episode {episode_id}")
            return []

        # Find the speaker with the maximum time
        host_speaker = None
        max_time = -1
        for speaker, time_spoken in speaker_times.items():
            if time_spoken > max_time:
                max_time = time_spoken
                host_speaker = speaker
            logger.debug(f"Episode {episode_id}: Speaker {speaker} spoke for {time_spoken:.1f}s")


        if host_speaker:
            logger.info(f"Episode {episode_id}: Host identified as '{host_speaker}' "
                       f"(speaking time: {speaker_times[host_speaker]:.1f}s)")
        else:
             logger.warning(f"Episode {episode_id}: Could not determine host speaker.")
             # Fallback: if no host, maybe all are guests or unknown? Or return empty?
             # Returning empty seems safer if the primary logic fails.
             return []


        # Create labeled utterances
        utterances = []
        for segment in segments:
            # Re-validate segment before creating Utterance
            if not isinstance(segment, dict):
                 continue
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start")
            end = segment.get("end")
            text = segment.get("text", "").strip()

            # Re-validate timing and text
            if start is None or end is None or start < 0 or end < 0 or start >= end or not text:
                 logger.debug(f"Skipping segment for utterance creation due to validation fail: {segment}")
                 continue


            speaker_label = "Host" if speaker == host_speaker else "Guest"

            utterance = Utterance(
                episode_id=episode_id,
                start=start,
                end=end,
                speaker=speaker_label,
                text=text
            )

            utterances.append(utterance)

        logger.info(f"Episode {episode_id}: Created {len(utterances)} labeled utterances")
        return utterances
