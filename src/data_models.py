#!/usr/bin/env python3
"""
Data models for Arabic Podcast Transcription Pipeline
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class Utterance:
    """Individual utterance with speaker and timing information"""
    episode_id: str
    start: float
    end: float
    speaker: str
    text: str
    
    def __post_init__(self):
        """Validate utterance data after initialization"""
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        if not self.text.strip():
            raise ValueError("Text cannot be empty")
    
    @property
    def duration(self) -> float:
        """Get duration of the utterance in seconds"""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert utterance to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert utterance to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class TranscriptionSegment:
    """Transcription segment with timing and speaker information"""
    start: float
    end: float
    text: str
    speaker: str
    words: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Validate segment data"""
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
    
    @property
    def duration(self) -> float:
        """Get duration of the segment in seconds"""
        return self.end - self.start
    
    def has_words(self) -> bool:
        """Check if segment has word-level information"""
        return self.words is not None and len(self.words) > 0


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata"""
    segments: List[TranscriptionSegment]
    detected_language: str = "ar"
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    
    def __post_init__(self):
        """Validate transcription result"""
        if not isinstance(self.segments, list):
            raise ValueError("Segments must be a list")
    
    @property
    def total_duration(self) -> float:
        """Get total duration of all segments"""
        if not self.segments:
            return 0.0
        return max(segment.end for segment in self.segments)
    
    @property
    def segment_count(self) -> int:
        """Get number of segments"""
        return len(self.segments)
    
    def get_speakers(self) -> List[str]:
        """Get list of unique speakers"""
        return list(set(segment.speaker for segment in self.segments))


@dataclass
class EpisodeMetadata:
    """Metadata for a podcast episode"""
    episode_id: str
    file_path: str
    file_size: int
    duration: Optional[float] = None
    processed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    utterance_count: int = 0
    speakers: Optional[List[str]] = None
    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.processed_at is None:
            self.processed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        data = asdict(self)
        if self.processed_at:
            data['processed_at'] = self.processed_at.isoformat()
        return data


@dataclass
class ProcessingError:
    """Error information for failed processing"""
    episode_id: str
    error_type: str
    error_message: str
    stage: str
    timestamp: datetime
    
    def __post_init__(self):
        """Set default timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary"""
        return {
            'episode_id': self.episode_id,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stage': self.stage,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert error to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class SpeakerStats:
    """Statistics for a speaker in an episode"""
    speaker_id: str
    speaker_label: str
    total_speaking_time: float
    utterance_count: int
    avg_utterance_duration: float
    
    @classmethod
    def from_utterances(cls, utterances: List[Utterance], speaker_id: str, speaker_label: str) -> 'SpeakerStats':
        """Create speaker statistics from utterances"""
        speaker_utterances = [u for u in utterances if u.speaker == speaker_label]
        
        if not speaker_utterances:
            return cls(
                speaker_id=speaker_id,
                speaker_label=speaker_label,
                total_speaking_time=0.0,
                utterance_count=0,
                avg_utterance_duration=0.0
            )
        
        total_time = sum(u.duration for u in speaker_utterances)
        count = len(speaker_utterances)
        avg_duration = total_time / count if count > 0 else 0.0
        
        return cls(
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            total_speaking_time=total_time,
            utterance_count=count,
            avg_utterance_duration=avg_duration
        )


@dataclass
class EpisodeResult:
    """Complete result for a processed episode"""
    metadata: EpisodeMetadata
    utterances: List[Utterance]
    speaker_stats: List[SpeakerStats]
    errors: List[ProcessingError]
    
    @property
    def success(self) -> bool:
        """Check if episode was processed successfully"""
        return len(self.utterances) > 0 and len(self.errors) == 0
    
    @property
    def total_duration(self) -> float:
        """Get total duration from utterances"""
        if not self.utterances:
            return 0.0
        return max(u.end for u in self.utterances)
    
    def get_speaker_labels(self) -> List[str]:
        """Get list of speaker labels"""
        return [stats.speaker_label for stats in self.speaker_stats]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary"""
        return {
            'episode_id': self.metadata.episode_id,
            'success': self.success,
            'utterance_count': len(self.utterances),
            'total_duration': self.total_duration,
            'speakers': self.get_speaker_labels(),
            'processing_time': self.metadata.processing_time,
            'errors': len(self.errors)
        }


# Utility functions for data validation
def validate_segment_data(segment_data: Dict[str, Any]) -> bool:
    """Validate raw segment data from transcription API"""
    required_fields = ['start', 'end', 'text']
    
    # Check required fields
    if not all(field in segment_data for field in required_fields):
        return False
    
    # Check data types and values
    try:
        start = float(segment_data['start'])
        end = float(segment_data['end'])
        text = str(segment_data['text']).strip()
        
        if start < 0 or end <= start or not text:
            return False
            
        return True
    except (ValueError, TypeError):
        return False


def validate_word_data(word_data: Dict[str, Any]) -> bool:
    """Validate word-level data from transcription API"""
    required_fields = ['start', 'end', 'word']
    
    # Check required fields
    if not all(field in word_data for field in required_fields):
        return False
    
    # Check data types and values
    try:
        start = float(word_data['start'])
        end = float(word_data['end'])
        word = str(word_data['word']).strip()
        
        if start < 0 or end <= start or not word:
            return False
            
        return True
    except (ValueError, TypeError):
        return False