# Arabic Podcast Pipeline

A comprehensive pipeline for processing Arabic podcasts with automatic transcription, speaker diarization, and speaker labeling.

## Features

- **Audio Preprocessing**: Converts audio files to optimal format for transcription
- **Speaker Diarization**: Automatically identifies different speakers in the audio
- **Arabic Transcription**: Uses WhisperX model optimized for Arabic language
- **Speaker Labeling**: Automatically labels speakers as Host/Guest based on speaking time
- **Batch Processing**: Processes multiple episodes efficiently
- **Error Handling**: Comprehensive error handling and logging
- **Resume Support**: Can resume processing from where it left off

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- HuggingFace API token
- Replicate API token
- Google Colab or similar environment with sufficient storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd arabic-podcast-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure FFmpeg is installed:
```bash
# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# On macOS with Homebrew
brew install ffmpeg

# On Windows, download from https://ffmpeg.org/download.html
```

## Configuration

1. Update `config.py` with your API tokens:
```python
HF_TOKEN = "your_huggingface_token_here"
REPLICATE_TOKEN = "your_replicate_token_here"
```

2. Set your input and output directories:
```python
INPUT_DIR = "/path/to/your/audio/files"
OUTPUT_DIR = "/path/to/output/directory"
```

## Usage

### Basic Usage

```bash
python main.py
```

### API Tokens

Get your API tokens from:
- **HuggingFace**: https://huggingface.co/settings/tokens
- **Replicate**: https://replicate.com/account/api-tokens

### Input Files

Place your audio files in the `INPUT_DIR`. Supported formats:
- MP3
- WAV
- M4A
- FLAC
- AAC

### Output

The pipeline generates:
- `episodes.jsonl`: Processed utterances with speaker labels
- `errors.log`: Error log for debugging
- `temp/`: Temporary processed audio files (automatically cleaned)

### Output Format

Each line in `episodes.jsonl` contains:
```json
{
  "episode_id": "episode_name",
  "start": 12.5,
  "end": 18.2,
  "speaker": "Host",
  "text": "أهلاً وسهلاً بكم في هذه الحلقة الجديدة"
}
```

## Architecture

```
arabic-podcast-pipeline/
├── main.py              # Main pipeline orchestrator
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── src/
│   ├── audio_processor.py    # Audio preprocessing
│   ├── file_uploader.py      # File upload services
│   ├── transcriber.py        # Transcription with diarization
│   ├── speaker_labeler.py    # Speaker labeling logic
│   ├── data_models.py        # Data structures
│   └── utils.py             # Utility functions
└── docs/
    └── API.md               # API documentation
```

## Processing Pipeline

1. **Audio Preprocessing**: Converts to mono WAV 16kHz with normalization
2. **File Upload**: Uploads to public hosting service for API access
3. **Transcription**: Uses Replicate WhisperX for Arabic transcription with diarization
4. **Speaker Labeling**: Identifies Host/Guest based on speaking time
5. **Output Generation**: Saves structured results to JSONL format

## Error Handling

- Comprehensive logging at all stages
- Automatic retry mechanisms for network operations
- Graceful handling of corrupted audio files
- Detailed error logs for debugging

## Performance Considerations

- Processes files sequentially to avoid API rate limits
- Automatic cleanup of temporary files
- Memory-efficient processing for large batches
- Adaptive polling for long transcription jobs

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg and ensure it's in PATH
2. **API token errors**: Verify tokens are correct and have proper permissions
3. **Upload failures**: Check internet connection and file sizes
4. **Transcription timeouts**: Increase timeout values in config.py

### Debug Mode

Enable debug logging:
```python
LOGGING_LEVEL = "DEBUG"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WhisperX for Arabic speech recognition
- Replicate for hosting the transcription model
- HuggingFace for speaker diarization models

## Support

For issues and questions:
1. Check the error logs in `errors.log`
2. Review the troubleshooting section
3. Open an issue on the repository
