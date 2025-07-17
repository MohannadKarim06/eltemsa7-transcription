# 🎙️ Eltemsa7 Transcription Pipeline

This project downloads show episodes (from YouTube), transcribes them using **Replicate WhisperX**, labels speakers (Host/Guest), and saves the results in a structured format.

---

## 📁 Project Structure

.
├── config.py # Configuration for input/output dirs and API tokens
├── main.py # Entry point to run the full pipeline
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── src/ # Core source code
│ ├── audio_downloader.py
│ ├── audio_processor.py
│ ├── data_models.py
│ ├── file_uploader.py
│ ├── speaker_labeler.py
│ ├── transcriber.py
│ └── utils.py
└── docs/
└── API.md # API-level usage documentation



---

## 🚀 How to Set Up

### 1. Clone or Extract the Project

```bash
git clone <repo-url>  
cd podcast-pipeline/
2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash

pip install -r requirements.txt
Also ensure you have FFmpeg installed:

bash

ffmpeg -version
If not, install it:

Ubuntu: sudo apt install ffmpeg

Windows/macOS: https://ffmpeg.org/download.html

🔑 Add Your API Keys
Edit the config.py file and fill in:


HF_TOKEN = "your_huggingface_token"
REPLICATE_TOKEN = "your_replicate_token"
Get your tokens from:

HuggingFace: https://huggingface.co/settings/tokens

Replicate: https://replicate.com/account/api-tokens

▶️ How to Run
Option 1: Download + Transcribe

python main.py
This will:

Download audio from the default YouTube playlist

Process, transcribe, and diarize it

Save output to /content/processed/

Option 2: Re-run Using Existing Audio
If you've already downloaded audio and want to reprocess:


python main.py --skip-download
(Alternatively, modify main() or call pipeline.run_full_pipeline(skip_download=True) manually.)

📂 Output Files
processed/episodes.jsonl — Transcribed utterances with speaker labels

processed/errors.log — Errors encountered

processed/download.log — Audio download logs

📄 API Details
See docs/API.md for developer-facing usage and class descriptions.
