# API Documentation

## Overview

This document describes the internal API structure and interfaces for the Arabic Podcast Pipeline.

## Core Classes

### AudioProcessor

Handles audio preprocessing and normalization.

```python
class AudioProcessor:
    def __init__(self, target_sr: int = 16000)
    def preprocess_audio(self, input_path: str, output_path: str) -> bool
```

**Methods:**
- `preprocess_audio()`: Converts audio to mono WAV 16kHz format with normalization

**Parameters:**
- `target_sr`: Target sample rate (default: 16000)

**Returns:**
- `bool`: True if successful, False otherwise

### FileUploader

Manages file uploads to various hosting services.

```python
class FileUploader:
    def __init__(self)
    def upload_file(self, file_path: str, max_retries: int = 3) -> Optional[str]
```

**Methods:**
- `upload_file()`: Uploads file and returns public URL

**Supported Services:**
- transfer.sh (14 days expiry)
- pomf.lain.la (72 hours expiry)
- uguu.se (24 hours expiry)
- catbox.moe (permanent)
- tmpfiles.org (1 hour expiry)
- 0x0.st (365 days expiry)

**Parameters:**
- `file_path`: Path to file to upload
- `max_retries`: Maximum retry attempts per service

**Returns:**
- `Optional[str]`: Public URL if successful, None otherwise

### ReplicateTranscriber

Handles transcription using Replicate's WhisperX model.

```python
class ReplicateTranscriber:
    def __init__(self, hf_token: str, replicate_token: str, model_version: str, 
                 base_url: str, blend_threshold: float = 2.0)
    def transcribe_with_diarization(self, audio_url: str, max_retries: int = 3) -> Dict[str, Any]
```

**Methods:**
- `transcribe_with_diarization()`: Transcribes audio with speaker diarization

**Parameters:**
- `audio_url`: Public URL of audio file
- `max_retries`: Maximum retry attempts

**Returns:**
- `Dict[str, Any]`: Transcription result with segments

### SpeakerLabeler

Labels speakers as Host/Guest based on speaking time.

```python
class SpeakerLabeler:
    def label_speakers(self, segments: List[Dict[str, Any]], episode_id: str) -> List[Utterance]
```

**Methods:**
- `label_speakers()`: Processes segments and assigns Host/Guest labels

**Parameters:**
- `segments`: List of transcription segments
- `episode_id`: Unique identifier for the episode

**Returns:**
- `List[Utterance]`: List of labeled utterances

## Data Models

### Utterance

Core data structure for processed speech segments.

```python
@dataclass
class Utterance:
    episode_id: str
    start: float
    end: float
    speaker: str
    text: str
```

**Fields:**
- `episode_id`: Unique identifier for the episode
- `start`: Start time in seconds
- `end`: End time in seconds
- `speaker`: Speaker label ("Host" or "Guest")
- `text`: Transcribed text content

## Configuration

### Config Parameters

```python
# Directories
INPUT_DIR: str          # Input audio files directory
OUTPUT_DIR: str         # Output directory for results
TEMP_DIR: str          # Temporary files directory

# API Tokens
HF_TOKEN: str          # HuggingFace API token
REPLICATE_TOKEN: str   # Replicate API token

# Processing Parameters
TARGET_AUDIO_SR: int        # Target sample rate (16000)
BLEND_THRESHOLD: float      # Utterance blending threshold (2.0)
UPLOADER_MAX_RETRIES: int  # Upload retry attempts (3)
POLLING_MAX_WAIT_TIME: int # Max transcription wait time (1800s)
POLLING_INTERVAL: int      # Polling interval (15s)

# Model Configuration
REPLICATE_MODEL_VERSION: str  # Model version hash
REPLICATE_BASE_URL: str      # Replicate API base URL
REPLICATE_TIMEOUT: int       # Request timeout (1200s)
```

## Pipeline Flow

### 1. Audio Preprocessing

```
Input Audio → FFmpeg Processing → Normalized WAV
```

**Process:**
- Convert to mono
- Resample to 16kHz
- Apply loudness normalization
- Save as WAV format

### 2. File Upload

```
Local WAV → Upload Service → Public URL
```

**Services Priority:**
1. transfer.sh (most reliable)
2. pomf.lain.la
3. uguu.se
4. catbox.moe
5. tmpfiles.org
6. 0x0.st (fallback)

### 3. Transcription

```
Public URL → Replicate API → Transcription + Diarization
```

**Parameters:**
- Language: Arabic ("ar")
- Diarization: Enabled
- Alignment: Enabled
- Batch size: 32
- Temperature: 0.0

### 4. Speaker Labeling

```
Raw Segments → Speaker Analysis → Host/Guest Labels
```

**Logic:**
- Calculate speaking time per speaker
- Assign "Host" to speaker with most time
- Assign "Guest" to other speakers

### 5. Output Generation

```
Labeled Utterances → JSONL Format → File Save
```

**Format:**
- One JSON object per line
- UTF-8 encoding
- Chronological order

## Error Handling

### Error Types

1. **Configuration Errors**
   - Missing API tokens
   - Invalid directory paths
   - Missing dependencies

2. **Audio Processing Errors**
   - Corrupted audio files
   - Unsupported formats
   - FFmpeg failures

3. **Upload Errors**
   - Network connectivity issues
   - Service unavailability
   - File size limitations

4. **Transcription Errors**
   - API rate limits
   - Service timeouts
   - Invalid responses

5. **Data Processing Errors**
   - Malformed transcription data
   - Missing speaker information
   - JSON serialization issues

### Error Recovery

- Automatic retries with exponential backoff
- Service fallbacks for uploads
- Graceful degradation
- Comprehensive logging

## Usage Examples

### Basic Pipeline Usage

```python
from main import PodcastPipeline
import config

# Update configuration
config.HF_TOKEN = "your_hf_token"
config.REPLICATE_TOKEN = "your_replicate_token"
config.INPUT_DIR = "/path/to/audio/files"
config.OUTPUT_DIR = "/path/to/output"

# Create and run pipeline
pipeline = PodcastPipeline()
pipeline.run_pipeline()
```

### Single Episode Processing

```python
from pathlib import Path

pipeline = PodcastPipeline()
episode_path = Path("/path/to/episode.mp3")
success = pipeline.process_episode(episode_path)
```

### Custom Configuration

```python
import config

# Override default settings
config.BLEND_THRESHOLD = 3.0
config.UPLOADER_MAX_RETRIES = 5
config.POLLING_MAX_WAIT_TIME = 3600
config.LOGGING_LEVEL = "DEBUG"
```

## Performance Considerations

### Memory Usage

- Audio files are processed one at a time
- Temporary files are cleaned up automatically
- Memory is freed after each episode

### Processing Time

- Depends on audio length and complexity
- Typical processing: 5-10x real-time
- Network latency affects upload/transcription

### Rate Limits

- Respects API rate limits
- Sequential processing to avoid throttling
- Automatic backoff on errors

## Testing

### Unit Tests

```python
# Test audio processor
processor = AudioProcessor()
assert processor.preprocess_audio("input.mp3", "output.wav")

# Test file uploader
uploader = FileUploader()
url = uploader.upload_file("test.wav")
assert url is not None

# Test speaker labeler
labeler = SpeakerLabeler()
utterances = labeler.label_speakers(segments, "episode_1")
assert len(utterances) > 0
```

### Integration Tests

```python
# Test full pipeline
pipeline = PodcastPipeline()
success = pipeline.process_episode(Path("test_episode.mp3"))
assert success == True

# Verify output
with open("episodes.jsonl", "r") as f:
    lines = f.readlines()
    assert len(lines) > 0
```

## Security Considerations

- API tokens are stored in configuration files
- Temporary files are cleaned up automatically
- No sensitive data is logged
- Network requests use HTTPS

## Future Enhancements

1. **Multi-language Support**
   - Automatic language detection
   - Support for other Arabic dialects

2. **Advanced Speaker Recognition**
   - Speaker identification across episodes
   - Voice embeddings for consistency

3. **Quality Improvements**
   - Confidence scoring
   - Manual correction interface
   - Quality metrics

4. **Performance Optimizations**
   - Parallel processing
   - Batch API calls
   - Caching mechanisms
