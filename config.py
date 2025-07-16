#!/usr/bin/env python3
"""
Configuration management for Arabic Podcast Transcription Pipeline
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    target_sample_rate: int = 16000
    format: str = "wav"
    codec: str = "pcm_s16le"
    channels: int = 1  # Mono
    min_duration: float = 10.0  # Minimum duration in seconds
    max_file_size_mb: float = 500.0  # Maximum file size in MB


@dataclass
class TranscriptionConfig:
    """Configuration for transcription settings"""
    model_version: str = "84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
    language: str = "ar"
    diarization: bool = True
    align_output: bool = True
    batch_size: int = 32
    temperature: float = 0.0
    debug: bool = False
    max_retries: int = 3
    timeout: int = 1200  # 20 minutes
    max_wait_time: int = 1800  # 30 minutes for polling


@dataclass
class SpeakerConfig:
    """Configuration for speaker labeling"""
    blend_threshold: float = 2.0  # Seconds to blend utterances
    min_speaking_time: float = 1.0  # Minimum speaking time to consider


@dataclass
class UploadConfig:
    """Configuration for file upload services"""
    max_retries: int = 3
    timeout: int = 300  # 5 minutes
    test_timeout: int = 30  # 30 seconds for URL testing


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    input_dir: str
    output_dir: str
    hf_token: str
    replicate_token: str
    
    # Sub-configurations
    audio: AudioConfig
    transcription: TranscriptionConfig
    speaker: SpeakerConfig
    upload: UploadConfig
    
    # File patterns
    audio_patterns: List[str]
    
    # Output files
    episodes_file: str = "episodes.jsonl"
    errors_file: str = "errors.log"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self):
        """Validate configuration values"""
        if not self.hf_token or self.hf_token == "your_huggingface_token_here":
            raise ValueError("Valid HuggingFace token required")
        
        if not self.replicate_token or self.replicate_token == "your_replicate_token_here":
            raise ValueError("Valid Replicate token required")
        
        if not Path(self.input_dir).exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def load_config(
    input_dir: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    replicate_token: Optional[str] = None,
    **kwargs
) -> PipelineConfig:
    """
    Load configuration with environment variable fallbacks
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory for output files
        hf_token: HuggingFace API token
        replicate_token: Replicate API token
        **kwargs: Additional configuration overrides
    
    Returns:
        PipelineConfig: Configured pipeline settings
    """
    # Get tokens from environment if not provided
    hf_token = hf_token or os.getenv("HF_TOKEN")
    replicate_token = replicate_token or os.getenv("REPLICATE_TOKEN")
    
    # Default audio patterns
    audio_patterns = kwargs.get("audio_patterns", ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac"])
    
    # Create sub-configurations
    audio_config = AudioConfig(**kwargs.get("audio", {}))
    transcription_config = TranscriptionConfig(**kwargs.get("transcription", {}))
    speaker_config = SpeakerConfig(**kwargs.get("speaker", {}))
    upload_config = UploadConfig(**kwargs.get("upload", {}))
    
    return PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        hf_token=hf_token,
        replicate_token=replicate_token,
        audio=audio_config,
        transcription=transcription_config,
        speaker=speaker_config,
        upload=upload_config,
        audio_patterns=audio_patterns,
        **{k: v for k, v in kwargs.items() if k not in ["audio", "transcription", "speaker", "upload", "audio_patterns"]}
    )


# Default configuration values
DEFAULT_CONFIG = {
    "audio": {
        "target_sample_rate": 16000,
        "min_duration": 10.0,
        "max_file_size_mb": 500.0
    },
    "transcription": {
        "max_retries": 3,
        "timeout": 1200,
        "batch_size": 32,
        "temperature": 0.0
    },
    "speaker": {
        "blend_threshold": 2.0,
        "min_speaking_time": 1.0
    },
    "upload": {
        "max_retries": 3,
        "timeout": 300
    }
}