# Super Voice Auto Trainer 🎤

An iterative voice training tool that builds character voice models gradually from individual audio clips. Process shorter audio sources one at a time, build a persistent character library, and create high-quality voice models using Gemini AI and Fish Audio.

## Features

- 🔄 **Iterative Voice Building:** Build character voices gradually across multiple sessions
- 📁 **Persistent Character Library:** Automatically saves and organizes voice clips by character
- 🤖 **Gemini 2.5 Pro AI:** Advanced speaker detection with thinking mode for better accuracy
- 📝 **Flexible Instructions:** Single prompt for context, target speakers, and custom instructions
- 🎯 **Smart Target Selection:** CSV/array input support for multiple speakers
- 🔊 **Individual Source Processing:** Each audio file processed separately for better accuracy
- 🎵 **Background Music Removal:** Optional music separation for cleaner voice extraction
- 📊 **Structured Output:** Reliable JSON parsing with multiple fallback methods
- 🐟 **Multi-Clip Fish Models:** Trains voice models using all clips for each character
- 🎧 **Interactive Refinement:** Chat with Gemini to fine-tune voice segments
- 🧠 **Thinking Mode:** See Gemini's reasoning process during analysis
- 📱 **Beautiful Terminal UI:** Rich console interface with real-time progress

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

#### 🤖 Iterative Voice Building (Recommended)
```bash
# Process single sources to build character library gradually
poetry run voice-trainer train --use-gemini \
  "https://www.youtube.com/watch?v=VIDEO_ID"

# Process multiple sources individually (not combined)
poetry run voice-trainer train --use-gemini \
  "https://www.youtube.com/watch?v=VIDEO_ID1" \
  "https://www.youtube.com/watch?v=VIDEO_ID2" \
  "/path/to/local/audio.wav"

# With background music removal for better voice isolation
poetry run voice-trainer train --use-gemini --remove-music \
  "https://www.youtube.com/watch?v=VIDEO_ID"

# Mix YouTube videos and local files (each processed separately)
poetry run voice-trainer train --use-gemini \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  "/path/to/local/audio.wav"
```

#### 🔬 Traditional pyannote Workflow (GPU Recommended)
```bash
# Use pyannote for local processing (requires good GPU)
poetry run voice-trainer train \
  "https://www.youtube.com/watch?v=VIDEO_ID"

# With background music removal
poetry run voice-trainer train --remove-music \
  "https://www.youtube.com/watch?v=VIDEO_ID"
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

## New Iterative Workflow

### 🔄 **How It Works**

This tool now uses an **iterative approach** instead of processing everything at once:

**1. 📹 Individual Processing**
- Each audio source (YouTube URL, local file) is processed separately
- Shorter clips = better speaker accuracy and isolation
- No complex multi-video stitching or timing issues

**2. 🎯 Flexible Instructions**
- Single prompt for context, target speakers, and custom instructions
- Examples: "Harry Potter movie, extract Harry and Hermione"
- CSV support: "Ross, Rachel, Monica" or "['Character 1', 'Character 2']"

**3. 📁 Persistent Character Library**
- Characters saved in `character_library/` folder
- Clips automatically organized by character name
- Metadata tracking: clip count, duration, descriptions

**4. 🤖 Gemini 2.5 Pro Analysis**
- Advanced AI speaker detection with thinking mode
- Context-aware character recognition
- Structured JSON output with robust error handling

**5. 🐟 Multi-Clip Voice Models**
- Fish Audio trains models using ALL clips for each character
- Each new session adds more clips to existing characters
- Higher quality models from diverse voice samples

### 🎯 **Interactive Experience**

```
🎯 Analysis Instructions (Optional)
Examples:
  📺 Context: 'Harry Potter movie scene'
  🎯 Target speakers: 'Harry Potter, Hermione'
  📝 Combined: 'Friends episode, get Ross and Rachel voices'

Instructions: Harry Potter movie, extract Harry and Hermione

📝 Instructions: Harry Potter movie, extract Harry and Hermione
🎯 Detected targets: Harry, Hermione
🧠 Gemini is analyzing the audio...
🧐 I can detect 3 distinct voices in this Harry Potter clip
✅ Gemini found 3 speakers with high confidence

📚 Character Library:
┌─────────────┬───────┬──────────┬────────────┐
│ Character   │ Clips │ Duration │ Updated    │
├─────────────┼───────┼──────────┼────────────┤
│ Harry       │ 2     │ 45.2s    │ 2024-01-15 │
│ Hermione    │ 1     │ 23.1s    │ 2024-01-14 │
└─────────────┴───────┴──────────┴────────────┘

🎤 Speaker: Speaker 1 (Young male voice) → Harry Potter
📋 Existing character info:
   • Current clips: 2
   • Total duration: 45.2s
   • Description: Young wizard protagonist
Add new clip to existing character 'Harry Potter'? [Y/n]: y
✅ Will add to existing character: Harry Potter
```

### 🆚 **Comparison: Old vs New**

| **Old Multi-Video Approach** | **New Iterative Approach** |
|-------------------------------|----------------------------|
| ❌ Complex multi-video stitching | ✅ Simple individual processing |
| ❌ Long audio = confusion | ✅ Short clips = better accuracy |
| ❌ One-shot voice building | ✅ Gradual voice library building |
| ❌ Temporary character data | ✅ Persistent character storage |
| ❌ Single file per character | ✅ Multi-clip model training |

## Interactive Features

### 📝 **Flexible Instructions**
The new workflow uses a single, flexible instruction field for everything:

**Input Options:**
```bash
# Context only
"Harry Potter movie scene"

# Target speakers (CSV)
"Harry Potter, Hermione, Dumbledore"

# Target speakers (Array)  
"['Ross', 'Rachel', 'Monica']"

# Combined instructions
"Friends TV show episode, extract Ross and Rachel voices"

# Custom analysis
"Focus on clear dialogue, ignore background voices, prioritize emotional scenes"
```

### 📁 **Character Library Management**
Build your voice library gradually across sessions:

**First Session:**
```bash
# Process a Harry Potter clip
Instructions: Harry Potter movie, extract Harry and Hermione

# Creates new characters:
✅ Created new character: Harry Potter
✅ Created new character: Hermione
```

**Later Sessions:**
```bash
# Process another clip with same characters
Instructions: Another Harry Potter scene, get Harry and Ron

# Adds to existing + creates new:
✅ Will add to existing character: Harry Potter  
✅ Created new character: Ron Weasley
```

### 💬 **Interactive Chat Refinement**
Fine-tune segments with natural language:

```
💬 Interactive Refinement Chat for Harry Potter
Current segments: 8 segments, 34.5s total

Chat: "Focus on the emotional parts, remove any laughing"
🤖 Gemini refined segments (high confidence)
💡 Kept emotional dialogue, removed 3 segments with laughter

Chat: "Make sure it's only Harry speaking, no overlapping voices"  
🤖 Gemini refined segments (high confidence)
💡 Removed 2 segments with background conversation

Chat: done
✅ Final segments: 5 segments, 28.1s total
```

### 🐟 **Multi-Clip Model Training**
Fish Audio automatically uses all clips for each character:

```
🐟 Creating Fish model for Harry Potter...
🎤 Used 4 audio clips:
  • 20241215_143022_Harry_Potter.wav (15.2s)
  • 20241215_150145_Harry_Potter.wav (28.1s) 
  • 20241214_162033_Harry_Potter.wav (12.8s)
  • 20241213_091247_Harry_Potter.wav (22.4s)
✅ Created model for Harry Potter: model_xyz123
```

## Best Practices for Iterative Voice Training

### 🎯 **Optimal Clip Selection**
- **Short clips work better:** 30 seconds to 3 minutes per source
- **Quality over quantity:** Better to have fewer perfect clips than many poor ones
- **Diverse contexts:** Mix dialogue, monologue, emotional states
- **Clear audio preferred:** Avoid heavy background noise or music

### 📈 **Building Character Libraries**
- **Start small:** Begin with 1-2 good clips per character
- **Add gradually:** Process new clips in separate sessions
- **Consistent naming:** Use exact same character names across sessions
- **Review before adding:** Check existing character info before confirming

### 🎵 **Audio Quality Tips**
- **Use `--remove-music`** for videos with background music
- **Target 30-60 seconds** total per character for training
- **Avoid overlapping speech** - Gemini will filter these out
- **Multiple takes are good** - Fish models improve with diverse samples

### ⚡ **Workflow Efficiency**
- **Process individually:** Don't try to batch everything at once
- **Use context instructions:** Help Gemini with show/movie context
- **Leverage character library:** Shows existing characters before assignment
- **Interactive refinement:** Use chat to fine-tune problem segments

## File Organization

The tool automatically organizes your voice training data:

```
project/
├── character_library/           # Persistent character storage
│   ├── characters.json         # Character metadata
│   ├── Harry_Potter/           # Character folder
│   │   └── clips/              # Individual voice clips
│   │       ├── 20241215_143022_segment.wav
│   │       ├── 20241215_150145_segment.wav
│   │       └── ...
│   ├── Hermione_Granger/
│   │   └── clips/
│   │       └── ...
│   └── ...
├── downloads/                   # Cached YouTube downloads
│   ├── video_0_abc123.wav
│   └── ...
└── speaker_outputs/            # Legacy output folder
```

### Character Library Benefits:
- **📁 Persistent Storage**: Characters survive across sessions
- **📊 Metadata Tracking**: Clip count, duration, source info
- **🔍 Easy Management**: View library status anytime
- **🐟 Multi-Clip Training**: Fish models use all clips automatically

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