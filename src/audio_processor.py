#!/usr/bin/env python3
"""
Audio preprocessing module for Arabic Podcast Transcription Pipeline
"""
import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .data_models import ProcessingError
from config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioInfo:
    """Audio file information"""
    duration: float
    sample_rate: int
    channels: int
    codec: str
    file_size: int
    
    def is_valid(self, config: AudioConfig) -> bool:
        """Check if audio meets minimum requirements"""
        return (
            self.duration >= config.min_duration and
            self.file_size > 0 and
            self.file_size < config.max_file_size_mb * 1024 * 1024
        )


class AudioProcessor:
    """
    Handles audio preprocessing and normalization for transcription
    
    Features:
    - Format conversion to WAV
    - Resampling to target sample rate
    - Channel reduction to mono
    - Audio normalization
    - Validation and quality checks
    """
    
    def __init__(self, config: AudioConfig):
        """
        Initialize audio processor
        
        Args:
            config: Audio processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_audio_info(self, file_path: str) -> Optional[AudioInfo]:
        """
        Get audio file information using ffprobe
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioInfo object or None if failed
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Audio file does not exist: {file_path}")
                return None
            
            # Use ffprobe to get file information
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"ffprobe failed: {result.stderr}")
                return None
            
            probe_data = json.loads(result.stdout)
            
            # Extract audio stream information
            audio_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                self.logger.error("No audio stream found in file")
                return None
            
            # Extract format information
            format_info = probe_data.get('format', {})
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            return AudioInfo(
                duration=float(format_info.get('duration', 0)),
                sample_rate=int(audio_stream.get('sample_rate', 0)),
                channels=int(audio_stream.get('channels', 0)),
                codec=audio_stream.get('codec_name', ''),
                file_size=file_size
            )
            
        except Exception as e:
            self.logger.error(f"Error getting audio info: {e}")
            return None
    
    def preprocess_audio(self, input_path: str, output_path: str) -> bool:
        """
        Preprocess audio file for transcription
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output processed file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate input file
            if not os.path.exists(input_path):
                self.logger.error(f"Input file does not exist: {input_path}")
                return False
            
            # Get audio information
            audio_info = self.get_audio_info(input_path)
            if not audio_info:
                self.logger.error("Failed to get audio information")
                return False
            
            # Validate audio meets requirements
            if not audio_info.is_valid(self.config):
                self.logger.error(f"Audio file does not meet requirements: "
                                f"duration={audio_info.duration:.2f}s, "
                                f"size={audio_info.file_size/(1024*1024):.2f}MB")
                return False
            
            self.logger.info(f"Processing audio: {input_path}")
            self.logger.info(f"  Duration: {audio_info.duration:.2f}s")
            self.logger.info(f"  Sample rate: {audio_info.sample_rate}Hz")
            self.logger.info(f"  Channels: {audio_info.channels}")
            self.logger.info(f"  Size: {audio_info.file_size/(1024*1024):.2f}MB")
            
            # Build ffmpeg command
            cmd = self._build_ffmpeg_command(input_path, output_path)
            
            # Execute conversion
            self.logger.info(f"Converting audio: {input_path} -> {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return False
            
            # Validate output file
            if not self._validate_output(output_path):
                return False
            
            self.logger.info(f"Audio preprocessing successful: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            return False
    
    def _build_ffmpeg_command(self, input_path: str, output_path: str) -> list:
        """
        Build ffmpeg command for audio preprocessing
        
        Args:
            input_path: Input file path
            output_path: Output file path
            
        Returns:
            List of command arguments
        """
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-i', input_path,
            '-ac', str(self.config.channels),  # Set channels (mono)
            '-ar', str(self.config.target_sample_rate),  # Set sample rate
            '-acodec', self.config.codec,  # Set audio codec
            '-f', self.config.format,  # Set output format
        ]
        
        # Add audio normalization filter
        # loudnorm filter provides better normalization than simple volume adjustment
        cmd.extend([
            '-filter:a', 'loudnorm=I=-16:TP=-1.5:LRA=11'
        ])
        
        cmd.append(output_path)
        
        return cmd
    
    def _validate_output(self, output_path: str) -> bool:
        """
        Validate the output audio file
        
        Args:
            output_path: Path to output file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(output_path):
                self.logger.error(f"Output file was not created: {output_path}")
                return False
            
            # Check file size
            output_size = os.path.getsize(output_path)
            if output_size < 1000:  # Less than 1KB is suspicious
                self.logger.error(f"Output file too small: {output_size} bytes")
                return False
            
            # Verify audio properties
            audio_info = self.get_audio_info(output_path)
            if not audio_info:
                self.logger.error("Failed to validate output audio properties")
                return False
            
            # Check if properties match configuration
            if audio_info.sample_rate != self.config.target_sample_rate:
                self.logger.warning(f"Sample rate mismatch: expected {self.config.target_sample_rate}, "
                                  f"got {audio_info.sample_rate}")
            
            if audio_info.channels != self.config.channels:
                self.logger.warning(f"Channel count mismatch: expected {self.config.channels}, "
                                  f"got {audio_info.channels}")
            
            self.logger.info(f"Output validation successful: {output_size} bytes")
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False
    
    def batch_preprocess(self, input_dir: str, output_dir: str, 
                        patterns: list = None) -> Dict[str, bool]:
        """
        Batch preprocess audio files in a directory
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory for output files
            patterns: List of file patterns to match (default: common audio formats)
            
        Returns:
            Dictionary mapping file paths to success status
        """
        if patterns is None:
            patterns = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac"]
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for pattern in patterns:
            audio_files.extend(input_path.glob(pattern))
        
        audio_files.sort()
        
        self.logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        results = {}
        for audio_file in audio_files:
            output_file = output_path / f"{audio_file.stem}_processed.wav"
            
            self.logger.info(f"Processing: {audio_file.name}")
            success = self.preprocess_audio(str(audio_file), str(output_file))
            results[str(audio_file)] = success
            
            if success:
                self.logger.info(f"✅ Successfully processed: {audio_file.name}")
            else:
                self.logger.error(f"❌ Failed to process: {audio_file.name}")
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Batch processing complete: {successful}/{len(results)} files processed")
        
        return results
    
    def cleanup_temp_files(self, temp_dir: str, keep_recent: int = 5):
        """
        Clean up temporary audio files
        
        Args:
            temp_dir: Directory containing temporary files
            keep_recent: Number of recent files to keep
        """
        try:
            temp_path = Path(temp_dir)
            if not temp_path.exists():
                return
            
            # Get all temp audio files
            temp_files = []
            for pattern in ["*.wav", "*.mp3", "*.m4a"]:
                temp_files.extend(temp_path.glob(pattern))
            
            if len(temp_files) <= keep_recent:
                return
            
            # Sort by modification time (oldest first)
            temp_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove oldest files
            files_to_remove = temp_files[:-keep_recent]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up temp file: {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    @staticmethod
    def check_ffmpeg_availability() -> bool:
        """
        Check if FFmpeg is available in the system
        
        Returns:
            True if FFmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def get_ffmpeg_version() -> Optional[str]:
        """
        Get FFmpeg version string
        
        Returns:
            Version string or None if not available
        """
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Extract version from first line
                first_line = result.stdout.split('\n')[0]
                if 'ffmpeg version' in first_line:
                    return first_line.split(' ')[2]
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None