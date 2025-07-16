import logging
from typing import List, Dict, Any

# Import Utterance dataclass
from .data_models import Utterance

# Configure logging
logger = logging.getLogger(__name__)

class SpeakerLabeler:
    """Labels speakers as Host/Guest based on speaking time"""

    def label_speakers(self, segments: List[Dict[str, Any]], episode_id: str) -> List[Utterance]:
        """Label speakers as Host/Guest based on total speaking time"""
        if not segments:
            return []

        # Calculate total speaking time per speaker
        speaker_times = {}
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            duration = end - start

            if duration > 0:  # Only count valid durations
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

        # Determine host as speaker with most speaking time
        if not speaker_times:
            logger.warning(f"No speaker times found for episode {{episode_id}}")
            return []

        host_speaker = max(speaker_times, key=speaker_times.get)
        logger.info(f"Episode {{episode_id}}: Host identified as {{host_speaker}} "
                   f"(speaking time: {{speaker_times[host_speaker]:.1f}}s)")

        # Log all speakers
        for speaker, time_spoken in speaker_times.items():
            logger.info(f"  {{speaker}}: {{time_spoken:.1f}}s")

        # Create labeled utterances
        utterances = []
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            speaker_label = "Host" if speaker == host_speaker else "Guest"

            text = segment.get("text", "").strip()
            if not text:  # Skip empty text
                continue

            utterance = Utterance(
                episode_id=episode_id,
                start=segment.get("start", 0),
                end=segment.get("end", 0),
                speaker=speaker_label,
                text=text
            )

            utterances.append(utterance)

        logger.info(f"Episode {{episode_id}}: Created {{len(utterances)}} labeled utterances")
        return utterances