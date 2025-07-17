```markdown
# 📄 API Reference — Podcast Processing Pipeline

This document describes key classes and methods for using the pipeline in other scripts or projects.

---

## 🔧 Configuration (`config.py`)

```python
INPUT_DIR = "/content/downloaded_audio"
OUTPUT_DIR = "/content/processed"
HF_TOKEN = "your_huggingface_token"
REPLICATE_TOKEN = "your_replicate_token"
🚀 Entry Point — main.py
python
Copy
Edit
from main import main
main()
Runs the full pipeline, including download and processing.

To skip downloading:

pipeline = PodcastPipeline(INPUT_DIR, OUTPUT_DIR, HF_TOKEN, REPLICATE_TOKEN)
pipeline.run_full_pipeline(skip_download=True)
🔁 Core Pipeline — PodcastPipeline
PodcastPipeline.__init__(input_dir, output_dir, hf_token, replicate_token)
Initializes all components and sets up directories.

run_full_pipeline(playlist_url=None, skip_download=False)
Runs the end-to-end process:

playlist_url: (optional) YouTube playlist URL

skip_download: Set to True to use existing audio files in INPUT_DIR

process_episode(path: Path)
Process a single audio file:

Cleans and normalizes audio

Uploads to a temp host

Transcribes with WhisperX

Labels speakers (Host/Guest)

Writes output to episodes.jsonl

🧠 Components
AudioDownloader
download_playlist(url)

download_single_video(url)

validate_url(url)

AudioProcessor
preprocess_audio(input_path, output_path)
Converts to 16kHz mono WAV and normalizes audio

FileUploader
upload_file(path)
Uploads to file-sharing services and returns a public URL

ReplicateTranscriber
transcribe_with_diarization(audio_url)
Calls Replicate’s WhisperX and processes the output into utterances

SpeakerLabeler
label_speakers(segments, episode_id)
Assigns "Host"/"Guest" labels based on speaking time

📦 Output Format
Each line in episodes.jsonl is a JSON object:

json
Copy
Edit
{
  "episode_id": "episode_001",
  "start": 12.3,
  "end": 15.8,
  "speaker": "Host",
  "text": "بسم الله الرحمن الرحيم وبه نستعين"
}
🧪 Testing Tips
You can run the pipeline on a single video:

pipeline.download_single_video("https://youtube.com/watch?v=...")  # downloads and logs
pipeline.run_processing_pipeline()  # processes all found audio files
❓ Need Help?
Contact the developer or open an issue in the project repository.
