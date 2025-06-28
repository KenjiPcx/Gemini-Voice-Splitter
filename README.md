# Super Voice Auto Trainer 🎤

A smart tool for training AI voice models from multiple YouTube videos and audio files. This tool automatically separates voices from audio across multiple sources, matches characters between videos, stitches their audio together with silence removal, and creates realistic voice models using Fish Audio AI.

## Features

- 🎬 **Multi-Video Processing:** Download audio from multiple YouTube URLs simultaneously
- 🤖 **AI-Powered Speaker Detection:** Choose between pyannote or Gemini AI for speaker separation
- 🎵 Remove background music for cleaner voice separation
- 🔊 Separate multiple voices with high accuracy
- 🔇 **Voice Activity Detection (VAD):** Automatic silence removal with 0.5s buffers
- 🔗 **Intelligent Audio Stitching:** Combine multiple sources before processing
- 🎧 Interactive voice preview and selection system  
- 🏷️ Smart labeling system with character suggestions
- 📊 **Structured Output:** Pydantic-based data models for reliable AI responses
- 🔍 **High-Confidence Refinement:** Second AI pass for quality assurance
- 🐟 Integration with Fish Audio API for model creation
- 🚀 Fast and efficient processing with GPU support (pyannote) or cloud AI (Gemini)
- 📊 Beautiful terminal UI with status displays
- 🧩 **Modular Architecture:** Clean, maintainable codebase with separate modules

## Installation

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone this repository:
```bash
git clone https://github.com/KenjiPcx/super-voice-auto-trainer.git
cd super-voice-auto-trainer
```

3. Install dependencies and create virtual environment:
```bash
poetry install
```

4. Install ffmpeg (required for audio processing):
```bash
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install ffmpeg

# On macOS:
brew install ffmpeg

# On Windows (with chocolatey):
choco install ffmpeg
```

5. (Optional but recommended) Install audio playback tools for voice previewing:
```bash
# On Ubuntu/Debian:
sudo apt-get install sox-fmt-all
# or
sudo apt-get install alsa-utils

# On macOS:
brew install sox

# This enables audio preview during voice selection and labeling
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

### 3. Gemini API Key (Optional - for AI-powered speaker separation)
Get your API key from Google AI Studio:

1. Go to https://aistudio.google.com/app/apikey
2. Create a new API key
3. This enables the advanced Gemini AI speaker detection workflow

### 4. Environment Variables
Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and add your credentials:
```bash
HF_TOKEN=your_huggingface_token_here
FISH_API_KEY=your_fish_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Optional for AI speaker detection
```

Alternatively, you can pass them as command-line options (see usage below).

## Usage

### Basic Usage

#### 🤖 Gemini AI Workflow (Recommended)
```bash
# Process videos with AI-powered speaker detection
poetry run voice-trainer train --use-gemini \
  "https://www.youtube.com/watch?v=VIDEO_ID1" \
  "https://www.youtube.com/watch?v=VIDEO_ID2"

# With background music removal
poetry run voice-trainer train --use-gemini --remove-music \
  "https://www.youtube.com/watch?v=VIDEO_ID1" \
  "/path/to/local/audio.wav"

# Mix YouTube videos and local files
poetry run voice-trainer train --use-gemini \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  "/path/to/local/audio.wav" \
  "/path/to/another/file.mp3"
```

#### 🔬 Traditional pyannote Workflow
```bash
# Process with pyannote speech separation (requires GPU for best performance)
poetry run voice-trainer train \
  "https://www.youtube.com/watch?v=VIDEO_ID1" \
  "https://www.youtube.com/watch?v=VIDEO_ID2"

# With background music removal
poetry run voice-trainer train --remove-music \
  "https://www.youtube.com/watch?v=VIDEO_ID1" \
  "https://www.youtube.com/watch?v=VIDEO_ID2"

# Skip music separation for faster iterations
poetry run voice-trainer train --skip-music-separation \
  "https://www.youtube.com/watch?v=VIDEO_ID1" \
  "https://www.youtube.com/watch?v=VIDEO_ID2"
```

#### 🧪 Testing & Utilities
```bash
# Test timestamp extraction with sample data
poetry run voice-trainer test-timestamps

# Test sound removal with sample timestamps
poetry run voice-trainer test-removal
```

### Using Poetry Shell

```bash
# Enter the Poetry environment once
poetry shell

# Process multiple videos in one session
voice-trainer --remove-music \
  "https://youtube.com/watch?v=VIDEO_ID1" \
  "https://youtube.com/watch?v=VIDEO_ID2" \
  "local_file.wav"

# Or run separate sessions
voice-trainer "https://youtube.com/watch?v=VIDEO_ID1"
voice-trainer "/path/to/another/audio.wav"
```

### With explicit credentials

```bash
poetry run voice-trainer \
  --hf-token "your_hf_token" \
  --fish-api-key "your_fish_key" \
  --remove-music \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Command Options

- `--use-gemini`: Use Gemini AI for speaker separation instead of pyannote (recommended)
- `--gemini-api-key`: Gemini API key (overrides .env file)
- `--remove-music`: Remove background music before voice separation (recommended for music videos/movies)
- `--separator-model`: Music separation model to use (default: UVR-MDX-NET-Inst_HQ_3.onnx)
- `--skip-music-separation`: Skip music separation step (useful if already processed)
- `--hf-token`: Hugging Face token (overrides .env file) - Required for pyannote
- `--fish-api-key`: Fish Audio API key (overrides .env file)

## AI Speaker Detection Workflows

### 🤖 Gemini AI Workflow (Recommended)

**Advantages:**
- ✅ **More Accurate:** Better speaker identification than traditional models
- ✅ **Descriptive:** Provides voice characteristics ("Deep male voice", "Excited female voice") 
- ✅ **High Confidence:** Two-pass refinement ensures only the best segments
- ✅ **No GPU Required:** Runs entirely through cloud API
- ✅ **Structured Output:** Pydantic models ensure reliable data parsing
- ✅ **Interactive:** Chat-like refinement process for each character

**Process:**
1. **🎬 Download & Combine** - All sources stitched together
2. **🎵 Music Removal** - Optional background music separation
3. **🎯 Target Selection** - Optionally specify which characters to extract
4. **🤖 Gemini Analysis** - AI identifies speakers with voice descriptions (targeted or all)
5. **🎭 Initial Extraction** - Creates speaker files from AI timestamps
6. **👤 Interactive Review** - User confirms/edits character names
7. **🔍 Gemini Refinement** - AI extracts only high-confidence segments
8. **💬 Interactive Chat** - Fine-tune segments with conversational feedback
9. **🐟 Model Creation** - Creates Fish Audio models with refined data

### 🔬 pyannote Workflow (Traditional)

**Advantages:**
- ✅ **Local Processing:** No API calls required
- ✅ **GPU Accelerated:** Fast processing with CUDA support
- ✅ **Established:** Proven speech separation technology

**Best For:**
- Users with powerful GPUs
- Offline processing requirements
- When Gemini API is unavailable

## Interactive Features

### 🎯 Targeted Speaker Extraction
When using Gemini AI, you can specify exactly which characters you want to extract:

```bash
# The tool will ask you to specify target speakers
poetry run voice-trainer train --use-gemini "movie.mp4"

# Example interaction:
# 🎯 Target Specific Speakers (Optional)
# Do you want to target specific speakers? [y/N]: y
# Enter speaker/character name: Harry Potter
# ✅ Added: Harry Potter
# Add another speaker? [y/N]: y  
# Enter speaker/character name: Hermione
# ✅ Added: Hermione
# 🎯 Will target: Harry Potter, Hermione
```

### 💬 Interactive Chat Refinement
After initial processing, you can chat with Gemini to fine-tune each character's audio segments:

**Example Chat Commands:**
- `"Focus on the emotional dialogue"`
- `"Remove segments with background noise"`  
- `"Keep only clear speech, no whispering"`
- `"Prioritize segments where the character sounds angry"`
- `"Remove parts where other characters are talking"`

**Chat Flow:**
```
💬 Interactive Refinement Chat for Harry Potter
📋 Current segments for Harry Potter (12 segments):
  1. 0:15 - 0:23
  2. 0:45 - 0:52
  ...
📊 Total duration: 45.2s

💬 Chat with Gemini (or 'done'/'restart'): Focus on emotional parts
🤖 Gemini is processing your request...
✅ Gemini refined segments (high)
💡 Explanation: Kept segments with strong emotional delivery, removed neutral dialogue
```

**Special Commands:**
- `done` - Finish refinement when satisfied
- `restart` - Reset to original timestamps
- Any natural language instruction for segment refinement

## Multi-Video Strategy

### Best Practices for Character Voice Training:

**🎯 Optimal Input Selection:**
- Use 3-5 videos per character for best results
- Choose videos with different speaking contexts (dialogue, monologue, emotional states)
- Prefer videos with clear audio and minimal background noise
- Mix short clips (30s-2min) for variety

**📊 Quality Tips:**
- Enable `--remove-music` for videos with background music
- Choose videos where target characters speak for at least 10-15 seconds total
- Avoid videos with heavy audio effects or distortion
- Use videos with multiple characters to train several voices simultaneously

**⚡ Efficiency Tips:**
- Process all related videos in one session to maintain character context
- Use consistent character naming across videos (case-sensitive)
- Preview audio before labeling to ensure voice quality
- Skip low-quality or very short voice segments

## Enhanced Multi-Video Workflow

### Smart Multi-Video Processing:
1. **🎬 Multi-Input**: Provide multiple YouTube URLs and/or local audio files
2. **📥 Batch Download**: All audio sources are downloaded/loaded first (with caching)
3. **🔇 VAD Pre-processing**: Remove silence/gaps from each video using Voice Activity Detection
4. **🔗 Intelligent Stitching**: Combine cleaned audio with 0.5s buffers between segments
5. **🎵 Music Removal** *(Optional/Skippable)*: Remove background music from combined audio
6. **🔊 Single Voice Separation**: Run voice separation once on the optimized combined audio
   - **📈 Detailed Progress**: Real-time progress tracking with pyannote ProgressHook
   - Shows processing stages, completion percentages, and timing estimates
7. **🎧 Interactive Selection**: Preview and choose which voice tracks to keep
   - Each track contains segments from ALL videos automatically
   - Play audio previews for each separated voice
   - Keep, discard, or skip each track
   - Beautiful terminal interface with tables and progress bars

### Final Processing:
8. **🏷️ Character Labeling**: Label the long voice tracks with character names
9. **🔇 Final VAD**: Additional silence removal using VAD for clean training audio
10. **🐟 Model Training**: Create Fish Audio models with full-length character audio
11. **📈 Results**: Get model IDs for high-quality character voices

### Progress Tracking Features:
- **Real-time Updates**: See exactly what pyannote is doing at each stage
- **No Conflicts**: Clean progress display without overlapping spinners
- **Timing Estimates**: Duration tracking for each processing step
- **Visual Feedback**: Color-coded status messages and progress indicators

### Key Advantages:
- ✅ **Full-Length Audio**: No automatic trimming - you get complete voice tracks
- ✅ **Smart Combination**: Automatically combines same voices across all videos
- ✅ **Single Processing**: Voice separation runs once on combined audio
- ✅ **Manual Control**: You decide the final length and trimming
- ✅ **Clean Audio**: Automatic silence removal and music separation

## Supported Formats

- **Input**: YouTube URLs, WAV, MP3, MP4, and most audio formats
- **Output**: Voice models hosted on Fish Audio platform

## Requirements

- Python 3.11+
- Hugging Face account with pyannote model access
- Fish Audio account with API access
- GPU recommended for faster processing
- **NEW:** WebRTC VAD for voice activity detection
- **NEW:** Audio Separator for music removal capabilities

## Troubleshooting

### Common Issues

1. **pyannote model access denied**: Make sure you've accepted the user conditions for both required models on Hugging Face.

2. **CUDA out of memory**: The tool will automatically fall back to CPU if GPU memory is insufficient.

3. **"Couldn't find ffmpeg" error**: Install ffmpeg using the instructions above. This is required for audio processing.

4. **No audio playback**: Install sox or alsa-utils for audio preview functionality.

5. **YouTube download fails**: Make sure yt-dlp is up to date: `pip install -U yt-dlp`

6. **SyntaxWarning from pydub**: These are harmless warnings from the pydub library and don't affect functionality.

### Performance Tips & Timing Estimates

**⏱️ Processing Times (approximate):**
- **YouTube Download**: 30s-2min per video
- **Music Separation**: 1-3min per minute of audio (CPU), 30s-1min (GPU)
- **Voice Separation**: 2-5min per minute of audio (CPU), 1-2min (GPU)
- **Voice Activity Detection**: 10-30s per minute of audio
- **Fish Audio Training**: 2-5min per model

**🚀 Optimization Tips:**
- Use GPU for faster processing (CUDA-compatible GPU required)
- **Multi-Video Mode**: Process 3-5 related videos per session for optimal character training
- Use shorter audio clips (5-10 minutes per video) for faster processing
- Ensure good audio quality in source material for better separation
- Enable `--remove-music` for videos with background music/sound effects

**📊 Total Time Estimates:**
- **Single 5-min video**: 10-20 minutes total
- **3 videos (multi-character)**: 30-60 minutes total
- **GPU vs CPU**: ~2-3x faster with GPU

### Multi-Video Specific Issues

5. **Character matching across videos**: Make sure to use identical character names (case-sensitive) when labeling voices from different videos.

6. **VAD (Voice Activity Detection) too aggressive**: If important speech is being removed, the VAD sensitivity is automatically adjusted. Very quiet speech might still be filtered out.

7. **Insufficient character audio**: If a character has less than 1 second of total audio after processing, the model creation will be skipped. This usually means the voice separation didn't work well.

8. **Memory issues with multiple videos**: Process fewer videos simultaneously or use shorter clips if you encounter memory problems.

## License

MIT License