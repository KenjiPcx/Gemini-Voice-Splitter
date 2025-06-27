# Super Voice Auto Trainer 🎤

A smart tool for training AI voice models from YouTube videos and audio files. This tool automatically separates voices from audio, lets you label them interactively, and creates realistic voice models using Fish Audio AI.

## Features

- 🎬 Download audio from YouTube URLs using yt-dlp
- 🔊 Separate multiple voices using pyannote speech separation
- ✂️ Automatically trim voice tracks to optimal length (45-60 seconds)
- 🏷️ Interactive labeling system for voice identification
- 🐟 Integration with Fish Audio API for model creation
- 🚀 Fast and efficient processing with GPU support

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd super-voice-auto-trainer
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install yt-dlp for YouTube downloads:
```bash
pip install yt-dlp
```

4. (Optional) Install audio playback tools for previewing separated voices:
```bash
# On Ubuntu/Debian:
sudo apt-get install sox-fmt-all
# or
sudo apt-get install alsa-utils

# On macOS:
brew install sox
```

## Setup

### 1. Hugging Face Token
You need a Hugging Face access token to use the pyannote model:

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with read permissions
3. Accept the user conditions for:
   - https://huggingface.co/pyannote/speech-separation-ami-1.0
   - https://huggingface.co/pyannote/segmentation-3.0

### 2. Fish Audio API Key
Get your API key from Fish Audio:

1. Sign up at https://fish.audio/
2. Get your API key from the dashboard

### 3. Environment Variables
Set your credentials as environment variables:

```bash
export HF_TOKEN="your_huggingface_token_here"
export FISH_API_KEY="your_fish_api_key_here"
```

Or pass them as command-line options (see usage below).

## Usage

### Basic Usage

```bash
# Process a YouTube video
python voice_trainer.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Process a local audio file
python voice_trainer.py "/path/to/your/audio.wav"
```

### With explicit credentials

```bash
python voice_trainer.py \
  --hf-token "your_hf_token" \
  --fish-api-key "your_fish_key" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Workflow

1. **Input**: Provide a YouTube URL or local audio file
2. **Download**: Audio is downloaded from YouTube (if URL provided)
3. **Separation**: Voices are separated using pyannote AI
4. **Trimming**: Each voice track is trimmed to 45-60 seconds
5. **Preview**: Listen to each separated voice track
6. **Labeling**: Interactively label each voice with name and description
7. **Training**: Create voice models on Fish Audio platform
8. **Results**: Get model IDs for your trained voices

## Supported Formats

- **Input**: YouTube URLs, WAV, MP3, MP4, and most audio formats
- **Output**: Voice models hosted on Fish Audio platform

## Requirements

- Python 3.8+
- Hugging Face account with pyannote model access
- Fish Audio account with API access
- GPU recommended for faster processing

## Troubleshooting

### Common Issues

1. **pyannote model access denied**: Make sure you've accepted the user conditions for both required models on Hugging Face.

2. **CUDA out of memory**: The tool will automatically fall back to CPU if GPU memory is insufficient.

3. **No audio playback**: Install sox or alsa-utils for audio preview functionality.

4. **YouTube download fails**: Make sure yt-dlp is up to date: `pip install -U yt-dlp`

### Performance Tips

- Use GPU for faster processing (CUDA-compatible GPU required)
- Use shorter audio clips (5-10 minutes) for faster processing
- Ensure good audio quality in source material for better separation

## License

MIT License