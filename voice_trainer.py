#!/usr/bin/env python3
"""
Super Voice Auto Trainer - A tool for training AI voice models from YouTube videos
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
import requests
import click
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
from dotenv import load_dotenv
from audio_separator.separator import Separator
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import webrtcvad
import numpy as np
from collections import defaultdict
import google.generativeai as genai
import json

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# Load environment variables from .env file
load_dotenv()

console = Console()


class VoiceTrainer:
    def __init__(self, hf_token: str, fish_api_key: str, remove_music: bool = False, separator_model: Optional[str] = None, skip_music_separation: bool = False, gemini_api_key: Optional[str] = None, use_gemini: bool = False):
        self.hf_token = hf_token
        self.fish_api_key = fish_api_key
        self.remove_music = remove_music
        self.separator_model = separator_model
        self.skip_music_separation = skip_music_separation
        self.gemini_api_key = gemini_api_key
        self.use_gemini = use_gemini
        self.pipeline = None
        self.separator = None
        self.temp_dir = None
        self.downloads_dir = os.path.join(os.getcwd(), "downloads")  # Persistent downloads
        self.character_voices = defaultdict(list)  # Store voice clips by character name
        
        # Create persistent downloads directory
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Setup Gemini if requested
        if self.use_gemini and self.gemini_api_key:
            self.setup_gemini()
    
    def setup_gemini(self):
        """Initialize Gemini API"""
        try:
            console.print("🤖 Setting up Gemini API...", style="cyan")
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            console.print("✅ Gemini API ready", style="green")
        except Exception as e:
            console.print(f"❌ Error setting up Gemini: {e}", style="red")
            self.use_gemini = False
    
    def analyze_audio_with_gemini(self, audio_file: str) -> Dict[str, List[Tuple[str, str]]]:
        """Use Gemini to analyze audio and extract speaker timestamps"""
        try:
            console.print("🤖 Analyzing audio with Gemini...", style="cyan")
            
            # Upload audio file to Gemini
            console.print("📤 Uploading audio to Gemini...", style="cyan")
            audio_file_obj = genai.upload_file(audio_file)
            
            prompt = """
            Please analyze this audio file and identify different speakers. For each distinct speaker you detect, provide:

            1. A brief description of their voice characteristics (e.g., "Deep male voice", "Higher-pitched female voice", "Child's voice", etc.)
            2. All the timestamps where that speaker is talking in MM:SS - MM:SS format

            Please be conservative and only include timestamps where you're confident about the speaker identity. Format your response as JSON like this:

            {
                "Speaker 1 (Deep male voice)": [
                    ["0:05", "0:12"],
                    ["0:25", "0:31"]
                ],
                "Speaker 2 (Higher female voice)": [
                    ["0:13", "0:24"],
                    ["0:32", "0:45"]
                ]
            }

            Only include speakers that have at least 3 seconds of total speaking time. Be precise with timestamps and conservative with speaker identification.
            """
            
            console.print("🧠 Gemini is analyzing the audio...", style="cyan")
            response = self.gemini_model.generate_content([audio_file_obj, prompt])
            
            # Parse JSON response
            try:
                # Extract JSON from response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                speaker_data = json.loads(response_text)
                
                # Convert to the format we need (list of tuples)
                formatted_data = {}
                for speaker, timestamps in speaker_data.items():
                    formatted_data[speaker] = [(start, end) for start, end in timestamps]
                
                console.print(f"✅ Gemini found {len(formatted_data)} speakers", style="green")
                return formatted_data
                
            except json.JSONDecodeError as e:
                console.print(f"❌ Error parsing Gemini response: {e}", style="red")
                console.print(f"Response was: {response.text[:200]}...", style="yellow")
                return {}
                
        except Exception as e:
            console.print(f"❌ Error analyzing audio with Gemini: {e}", style="red")
            return {}
    
    def refine_speaker_with_gemini(self, audio_file: str, speaker_name: str, description: str) -> List[Tuple[str, str]]:
        """Use Gemini to refine timestamps for a specific speaker"""
        try:
            console.print(f"🤖 Refining timestamps for [bold]{speaker_name}[/bold] with Gemini...", style="cyan")
            
            # Upload audio file to Gemini
            audio_file_obj = genai.upload_file(audio_file)
            
            prompt = f"""
            This audio clip should contain the voice of {speaker_name} ({description}). 

            Please analyze this audio and provide ONLY the timestamps where you are 100% confident that {speaker_name} is speaking clearly and audibly. Be very conservative - it's better to exclude uncertain segments than include wrong ones.

            Provide your response as a JSON array of timestamp pairs in MM:SS format:

            [
                ["0:02", "0:08"],
                ["0:15", "0:22"],
                ["0:35", "0:41"]
            ]

            Only include segments where:
            1. You're completely sure it's {speaker_name} speaking
            2. The audio quality is good
            3. There's minimal background noise or other speakers
            4. The segment is at least 1 second long

            If you're not confident about any segments, return an empty array [].
            """
            
            console.print("🧠 Gemini is refining the voice segments...", style="cyan")
            response = self.gemini_model.generate_content([audio_file_obj, prompt])
            
            # Parse JSON response
            try:
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                timestamps = json.loads(response_text)
                refined_timestamps = [(start, end) for start, end in timestamps]
                
                console.print(f"✅ Gemini refined to {len(refined_timestamps)} high-confidence segments", style="green")
                return refined_timestamps
                
            except json.JSONDecodeError as e:
                console.print(f"❌ Error parsing Gemini refinement response: {e}", style="red")
                console.print(f"Response was: {response.text[:200]}...", style="yellow")
                return []
                
        except Exception as e:
            console.print(f"❌ Error refining speaker with Gemini: {e}", style="red")
            return []
        
    def setup_pipeline(self):
        """Initialize the pyannote pipeline"""
        try:
            console.print("🔧 Setting up pyannote pipeline...", style="cyan")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speech-separation-ami-1.0",
                use_auth_token=self.hf_token
            )
            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                console.print("✅ Using GPU for processing", style="green")
            else:
                console.print("⚠️ Using CPU for processing", style="yellow")
        except Exception as e:
            console.print(f"❌ Error setting up pyannote pipeline: {e}", style="red")
            sys.exit(1)
    
    def setup_music_separator(self):
        """Initialize the music separator"""
        if not self.remove_music:
            return
            
        try:
            console.print("🎵 Setting up music separator...", style="cyan")
            # Initialize separator - newer version uses different API
            self.separator = Separator()
            if self.separator_model:
                console.print(f"🎵 Loading separator model: [bold]{self.separator_model}[/bold]...", style="cyan")
                self.separator.load_model(self.separator_model)
            console.print("✅ Music separator ready", style="green")
        except Exception as e:
            console.print(f"❌ Error setting up music separator: {e}", style="red")
            console.print("⚠️ Music separation will be disabled", style="yellow")
            self.separator = None
    
    def remove_background_music(self, audio_path: str) -> str:
        """Remove background music and return vocals-only audio"""
        if not self.remove_music or not self.separator or self.skip_music_separation:
            if self.skip_music_separation:
                console.print("⏭️ Skipping music separation (as requested)", style="yellow")
            return audio_path
            
        console.print("🎵 Removing background music...", style="cyan")
        
        try:
            # Create output directory for separated audio
            music_output_dir = os.path.join(self.temp_dir, "music_separation")
            os.makedirs(music_output_dir, exist_ok=True)
            
            # Separate vocals from music
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Separating vocals from music...", total=None)
                
                # Use default model for vocal separation
                output_files = self.separator.separate(audio_path)
                
                progress.update(task, completed=100)
            
            # Find the vocals file - audio-separator returns [vocals_path, instrumental_path]
            vocals_file = None
            if isinstance(output_files, (list, tuple)) and len(output_files) >= 1:
                vocals_file = output_files[0]  # First file is usually vocals
            elif isinstance(output_files, str):
                vocals_file = output_files
            
            if vocals_file and os.path.exists(vocals_file):
                console.print("✅ Successfully separated vocals from music", style="green")
                return vocals_file
            else:
                console.print("⚠️ Vocals file not found, using original audio", style="yellow")
                return audio_path
                
        except Exception as e:
            console.print(f"⚠️ Music separation failed: {e}", style="yellow")
            console.print("Continuing with original audio...", style="yellow")
            return audio_path
    
    def download_youtube_audio(self, url: str, video_index: int = 0) -> str:
        """Download audio from YouTube URL using yt-dlp with persistent storage"""
        import hashlib
        
        # Create a hash of the URL for unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"video_{video_index}_{url_hash}.wav"
        output_path = os.path.join(self.downloads_dir, filename)
        
        # Check if already downloaded
        if os.path.exists(output_path):
            console.print(f"✅ Using cached download: {filename}", style="green")
            # Copy to temp dir for processing
            temp_path = os.path.join(self.temp_dir, f"input_audio_{video_index}.wav")
            shutil.copy2(output_path, temp_path)
            return temp_path
        
        # Download to persistent location
        download_template = os.path.join(self.downloads_dir, f"video_{video_index}_{url_hash}.%(ext)s")
        
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "-o", download_template,
            url
        ]
        
        try:
            console.print(f"📥 Downloading audio from YouTube (saving to downloads/)...", style="cyan")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading...", total=None)
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                progress.update(task, completed=100)
            
            # Copy downloaded file to temp dir for processing
            if os.path.exists(output_path):
                console.print(f"✅ Audio downloaded and cached: {filename}", style="green")
                temp_path = os.path.join(self.temp_dir, f"input_audio_{video_index}.wav")
                shutil.copy2(output_path, temp_path)
                return temp_path
            else:
                raise FileNotFoundError("Downloaded audio file not found")
                
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Error downloading YouTube video: {e.stderr}", style="red")
            sys.exit(1)
    
    def separate_voices(self, audio_path: str) -> List[str]:
        """Separate voices using pyannote speech separation with progress tracking"""
        console.print("🗣️ Separating voices with detailed progress tracking...", style="cyan")
        
        try:
            # Use pyannote's built-in progress tracking (displays detailed progress)
            with ProgressHook() as hook:
                diarization, sources = self.pipeline(audio_path, hook=hook)
            
            separated_files = []
            
            # Extract speakers from diarization and corresponding sources
            speakers = list(diarization.labels())
            
            # Get the sample rate from the sources object (e.g., 16000)
            sample_rate = getattr(sources, 'sample_rate', 16000)
            
            for i, speaker in enumerate(speakers):
                try:
                    # Extract the source audio data for this speaker as a numpy array
                    source_data = sources.data[:, i]
                    
                    # Convert from float32 to 16-bit PCM for WAV format
                    source_data_16bit = (np.clip(source_data, -1.0, 1.0) * 32767).astype(np.int16)
                    
                    # Create an AudioSegment directly from the raw audio data in memory
                    source_audio = AudioSegment(
                        data=source_data_16bit.tobytes(),
                        sample_width=source_data_16bit.dtype.itemsize,  # Should be 2 for 16-bit audio
                        frame_rate=sample_rate,
                        channels=1  # pyannote sources are mono
                    )
                    
                    # Export the in-memory audio segment to its final WAV file
                    output_path = os.path.join(self.temp_dir, f"voice_{i}_{speaker}.wav")
                    source_audio.export(output_path, format="wav")
                    separated_files.append(output_path)
                    
                except Exception as e:
                    console.print(f"⚠️ Failed to process source for speaker {speaker} (track {i}): {e}", style="yellow")
                    continue
            
            console.print(f"✅ Found {len(separated_files)} voice tracks", style="green")
            return separated_files
            
        except Exception as e:
            console.print(f"❌ Error separating voices: {e}", style="red")
            return []
    
    def interactive_voice_selection(self, voice_files: List[str]) -> List[str]:
        """Interactive voice selection with preview and discard options"""
        if not voice_files:
            return []
            
        console.print("\n🎧 Voice Track Selection", style="bold blue")
        console.print("=" * 50)
        
        # Create table showing all tracks
        table = Table(title="Available Voice Tracks")
        table.add_column("Track #", style="cyan", no_wrap=True)
        table.add_column("File", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Action", style="yellow")
        
        selected_voices = []
        
        for i, voice_file in enumerate(voice_files):
            # Get duration
            try:
                audio = AudioSegment.from_file(voice_file)
                duration = len(audio) / 1000  # Convert to seconds
                duration_str = f"{duration:.1f}s"
            except:
                duration_str = "Unknown"
            
            table.add_row(
                f"{i+1}",
                os.path.basename(voice_file),
                duration_str,
                "Pending"
            )
        
        console.print(table)
        console.print("\n🎵 Audio Preview & Selection")
        console.print("Use the following options for each track:")
        console.print("• [green]keep[/green] - Keep this voice track")
        console.print("• [red]discard[/red] - Discard this voice track")  
        console.print("• [blue]play[/blue] - Play audio preview")
        console.print("• [yellow]skip[/yellow] - Skip for now")
        
        for i, voice_file in enumerate(voice_files):
            console.print(f"\n🎤 [bold]Track {i+1}[/bold]: {os.path.basename(voice_file)}")
            
            while True:
                action = Prompt.ask(
                    "Action",
                    choices=["keep", "discard", "play", "skip"],
                    default="keep"
                )
                
                if action == "play":
                    self._play_audio_preview(voice_file)
                    continue
                elif action == "keep":
                    selected_voices.append(voice_file)
                    console.print("✅ Track kept", style="green")
                    break
                elif action == "discard":
                    console.print("🗑️ Track discarded", style="red")
                    break
                elif action == "skip":
                    console.print("⏭️ Track skipped", style="yellow")
                    break
        
        console.print(f"\n📊 [bold]Selection Summary:[/bold] {len(selected_voices)} of {len(voice_files)} tracks kept")
        return selected_voices
    
    def _play_audio_preview(self, voice_file: str):
        """Play audio preview if possible"""
        try:
            console.print("🔊 Playing audio preview...", style="cyan")
            if shutil.which("play"):  # Sox play command
                subprocess.run(["play", voice_file], check=True, capture_output=True)
            elif shutil.which("aplay"):  # ALSA player
                subprocess.run(["aplay", voice_file], check=True, capture_output=True)
            elif shutil.which("afplay"):  # macOS player
                subprocess.run(["afplay", voice_file], check=True, capture_output=True)
            else:
                console.print("⚠️ Audio playback not available", style="yellow")
                console.print("Install: [bold]sox[/bold] (Linux/macOS) or [bold]alsa-utils[/bold] (Linux)")
        except Exception as e:
            console.print(f"⚠️ Could not play audio: {e}", style="yellow")
    
    def remove_silence_with_vad(self, audio: AudioSegment, aggressive_mode: int = 1) -> AudioSegment:
        """Remove silence from audio using Voice Activity Detection"""
        try:
            # Convert to the format expected by WebRTC VAD (16kHz, 16-bit, mono)
            vad_audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # Initialize VAD
            vad = webrtcvad.Vad(aggressive_mode)
            
            # Convert to raw audio data
            raw_audio = vad_audio.raw_data
            sample_rate = vad_audio.frame_rate
            
            # VAD works on 10, 20, or 30ms frames
            frame_duration = 30  # ms
            frame_size = int(sample_rate * frame_duration / 1000) * 2  # 2 bytes per sample
            
            voiced_frames = []
            current_segment = AudioSegment.empty()
            silence_buffer = AudioSegment.silent(duration=500)  # 0.5s buffer
            
            # Process audio in frames
            for i in range(0, len(raw_audio), frame_size):
                frame = raw_audio[i:i + frame_size]
                
                # Pad frame if necessary
                if len(frame) < frame_size:
                    frame = frame + b'\x00' * (frame_size - len(frame))
                
                # Check if frame contains speech
                try:
                    is_speech = vad.is_speech(frame, sample_rate)
                    
                    if is_speech:
                        # Add this frame to current segment
                        frame_audio = AudioSegment(
                            frame,
                            frame_rate=sample_rate,
                            sample_width=2,
                            channels=1
                        )
                        current_segment += frame_audio
                    else:
                        # If we have accumulated speech, save it with buffer
                        if len(current_segment) > 0:
                            voiced_frames.append(current_segment)
                            voiced_frames.append(silence_buffer)
                            current_segment = AudioSegment.empty()
                            
                except Exception:
                    # If VAD fails on this frame, assume it's speech
                    frame_audio = AudioSegment(
                        frame,
                        frame_rate=sample_rate,
                        sample_width=2,
                        channels=1
                    )
                    current_segment += frame_audio
            
            # Add final segment if it exists
            if len(current_segment) > 0:
                voiced_frames.append(current_segment)
            
            # Combine all voiced segments
            if voiced_frames:
                result = sum(voiced_frames)
                # Convert back to original format
                result = result.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
                console.print(f"🔇 Trimmed silence: {len(audio)/1000:.1f}s → {len(result)/1000:.1f}s", style="cyan")
                return result
            else:
                console.print("⚠️ No speech detected, keeping original audio", style="yellow")
                return audio
                
        except Exception as e:
            console.print(f"⚠️ VAD failed: {e}, keeping original audio", style="yellow")
            return audio
    
    def stitch_character_voices(self, character_name: str, voice_clips: List[AudioSegment]) -> AudioSegment:
        """Stitch together multiple voice clips for a character with silence removal"""
        if not voice_clips:
            return AudioSegment.empty()
        
        console.print(f"🔗 Stitching {len(voice_clips)} clips for [bold]{character_name}[/bold]", style="cyan")
        
        # Remove silence from each clip and add buffers
        processed_clips = []
        total_original_duration = 0
        total_processed_duration = 0
        
        for i, clip in enumerate(voice_clips):
            total_original_duration += len(clip)
            
            # Remove silence
            cleaned_clip = self.remove_silence_with_vad(clip)
            total_processed_duration += len(cleaned_clip)
            
            if len(cleaned_clip) > 5000:  # Only keep clips longer than 1 second
                processed_clips.append(cleaned_clip)
        
        if not processed_clips:
            console.print(f"⚠️ No valid clips found for {character_name} after processing", style="yellow")
            return AudioSegment.empty()
        
        # Combine all clips
        final_audio = sum(processed_clips)
        
        console.print(
            f"✅ [bold]{character_name}[/bold]: "
            f"{total_original_duration/1000:.1f}s → {total_processed_duration/1000:.1f}s "
            f"({len(processed_clips)} clips)",
            style="green"
        )
        
        return final_audio
    
    def interactive_labeling(self, voice_files: List[str], video_index: int = 0) -> List[Tuple[str, str, str]]:
        """Interactive labeling of selected voice tracks with character matching"""
        labeled_voices = []
        
        if not voice_files:
            console.print("❌ No voice tracks to label", style="red")
            return []
        
        console.print(f"\n🏷️ Voice Labeling - Video {video_index + 1} ({len(voice_files)} tracks)", style="bold blue")
        console.print("=" * 60)
        
        # Show existing characters if any
        if self.character_voices:
            console.print("\n📚 [bold]Existing Characters:[/bold]")
            for char_name, clips in self.character_voices.items():
                total_duration = sum(len(AudioSegment.from_file(clip)) for clip in clips) / 1000
                console.print(f"  • {char_name} ({len(clips)} clips, {total_duration:.1f}s total)")
        
        for i, voice_file in enumerate(voice_files):
            console.print(f"\n🎤 [bold]Track {i+1}[/bold]: {os.path.basename(voice_file)}")
            
            # Option to play audio again
            if Confirm.ask("🔊 Play audio preview?", default=False):
                self._play_audio_preview(voice_file)
            
            # Character name input with existing character suggestions
            if self.character_voices:
                console.print("💡 [dim]Existing characters: " + ", ".join(self.character_voices.keys()) + "[/dim]")
            
            name = Prompt.ask(
                "🏷️ Character/Speaker name (or 'skip' to skip)",
                default="",
            )
            
            if not name.strip() or name.lower() == 'skip':
                console.print("⏭️ Skipping track", style="yellow")
                continue
            
            name = name.strip()
            
            # Check if this is a new character or existing one
            if name in self.character_voices:
                console.print(f"🔗 Adding to existing character: [bold]{name}[/bold]", style="cyan")
                # Add to existing character's voice collection
                self.character_voices[name].append(voice_file)
                description = f"Voice of {name} (multi-clip)"
            else:
                console.print(f"✨ Creating new character: [bold]{name}[/bold]", style="green")
                # Get description for new character
                description = Prompt.ask(
                    "📝 Voice description",
                    default=f"Voice of {name}",
                )
                # Initialize new character's voice collection
                self.character_voices[name] = [voice_file]
            
            labeled_voices.append((voice_file, name, description.strip()))
            console.print(f"✅ Labeled as: [bold]{name}[/bold]", style="green")
        
        return labeled_voices
    
    def create_fish_model(self, audio_file: str, name: str, description: str) -> Optional[str]:
        """Create a model on Fish Audio platform"""
        url = "https://api.fish.audio/model"
        
        headers = {
            "Authorization": f"Bearer {self.fish_api_key}"
        }
        
        console.print(f"🐟 Creating Fish Audio model for: [bold]{name}[/bold]", style="cyan")
        
        # Prepare the form data
        with open(audio_file, 'rb') as f:
            files = {
                'voices': (f"{name}.wav", f, 'audio/wav')
            }
            data = {
                'title': name,
                'description': description,
                'train_mode': 'fast',
                'enhance_audio_quality': 'true'
            }
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Uploading and training model...", total=None)
                    response = requests.post(url, headers=headers, files=files, data=data)
                    progress.update(task, completed=100)
                
                response.raise_for_status()
                
                result = response.json()
                model_id = result.get('id')
                
                if model_id:
                    console.print(f"✅ Created model '[bold]{name}[/bold]' with ID: [green]{model_id}[/green]")
                    return model_id
                else:
                    console.print(f"❌ Failed to get model ID for '[bold]{name}[/bold]'", style="red")
                    return None
                    
            except requests.exceptions.RequestException as e:
                console.print(f"❌ Error creating model '[bold]{name}[/bold]': {e}", style="red")
                return None
    
    def process_single_source(self, input_source: str, video_index: int = 0) -> List[Tuple[str, str, str]]:
        """Process a single audio source and return labeled voices"""
        console.print(f"\n📹 [bold]Processing Video {video_index + 1}[/bold]: [blue]{input_source}[/blue]")
        
        # Download or copy audio
        if input_source.startswith(('http://', 'https://')):
            audio_path = self.download_youtube_audio(input_source, video_index)
        else:
            # Copy local file to temp directory
            filename = f"input_audio_{video_index}.wav"
            audio_path = os.path.join(self.temp_dir, filename)
            shutil.copy2(input_source, audio_path)
            console.print("✅ Audio file loaded", style="green")
        
        # Remove background music (if enabled)
        clean_audio_path = self.remove_background_music(audio_path)
        
        # Separate voices
        voice_files = self.separate_voices(clean_audio_path)
        
        if not voice_files:
            console.print(f"❌ No voice tracks found in video {video_index + 1}", style="red")
            return []
        
        # Interactive voice selection
        selected_voices = self.interactive_voice_selection(voice_files)
        
        if not selected_voices:
            console.print(f"❌ No voices selected from video {video_index + 1}", style="red")
            return []
        
        # Interactive labeling with character matching
        labeled_voices = self.interactive_labeling(selected_voices, video_index)
        
        return labeled_voices
    
    def create_final_character_models(self) -> List[str]:
        """Create final stitched audio for each character and train models"""
        if not self.character_voices:
            console.print("❌ No characters found", style="red")
            return []
        
        console.print(f"\n🎭 [bold]Creating Final Models for {len(self.character_voices)} Characters[/bold]", style="blue")
        console.print("=" * 60)
        
        model_ids = []
        
        for character_name, voice_files in self.character_voices.items():
            console.print(f"\n🎤 [bold]Processing {character_name}[/bold] ({len(voice_files)} clips)")
            
            # Load all audio clips for this character
            voice_clips = []
            for voice_file in voice_files:
                try:
                    audio = AudioSegment.from_file(voice_file)
                    voice_clips.append(audio)
                except Exception as e:
                    console.print(f"⚠️ Could not load {voice_file}: {e}", style="yellow")
            
            if not voice_clips:
                console.print(f"❌ No valid clips for {character_name}", style="red")
                continue
            
            # Stitch together the character's voice clips
            final_audio = self.stitch_character_voices(character_name, voice_clips)
            
            if len(final_audio) < 5000:  # Less than 5 seconds
                console.print(f"⚠️ Not enough audio for {character_name} ({len(final_audio)/1000:.1f}s)", style="yellow")
                continue
            
            # Save the final stitched audio
            final_audio_path = os.path.join(self.temp_dir, f"final_{character_name}.wav")
            final_audio.export(final_audio_path, format="wav")
            
            # Create Fish Audio model
            description = f"Multi-clip voice model for {character_name}"
            model_id = self.create_fish_model(final_audio_path, character_name, description)
            
            if model_id:
                model_ids.append(model_id)
        
        return model_ids
    
    def parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp in format MM:SS to seconds"""
        try:
            parts = timestamp_str.strip().split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            else:
                raise ValueError(f"Invalid timestamp format: {timestamp_str}")
        except ValueError as e:
            console.print(f"❌ Error parsing timestamp '{timestamp_str}': {e}", style="red")
            return 0.0
    
    def extract_audio_segments_from_timestamps(self, audio_file: str, timestamps: List[Tuple[str, str]], speaker_name: str) -> AudioSegment:
        """Extract and stitch audio segments based on timestamps"""
        try:
            # Load the source audio file
            audio = AudioSegment.from_file(audio_file)
            console.print(f"🎵 Loaded audio: {len(audio)/1000:.1f}s", style="cyan")
            
            segments = []
            total_extracted_duration = 0
            
            console.print(f"📋 Extracting {len(timestamps)} segments for [bold]{speaker_name}[/bold]", style="cyan")
            
            for start_str, end_str in timestamps:
                start_seconds = self.parse_timestamp(start_str)
                end_seconds = self.parse_timestamp(end_str)
                
                # Convert to milliseconds for pydub
                start_ms = int(start_seconds * 1000)
                end_ms = int(end_seconds * 1000)
                
                # Extract segment
                if start_ms < len(audio) and end_ms <= len(audio) and start_ms < end_ms:
                    segment = audio[start_ms:end_ms]
                    segments.append(segment)
                    duration = (end_ms - start_ms) / 1000
                    total_extracted_duration += duration
                    console.print(f"  ✅ {start_str} - {end_str} ({duration:.1f}s)", style="green")
                else:
                    console.print(f"  ⚠️ Invalid timestamp range: {start_str} - {end_str}", style="yellow")
            
            if not segments:
                console.print(f"❌ No valid segments extracted for {speaker_name}", style="red")
                return AudioSegment.empty()
            
            # Stitch segments together with small buffers
            console.print(f"🔗 Stitching {len(segments)} segments...", style="cyan")
            buffer = AudioSegment.silent(duration=500)  # 0.5 second buffer
            
            final_audio = segments[0]
            for segment in segments[1:]:
                final_audio += buffer + segment
            
            console.print(f"✅ [bold]{speaker_name}[/bold]: {total_extracted_duration:.1f}s extracted, {len(final_audio)/1000:.1f}s final", style="green")
            return final_audio
            
        except Exception as e:
            console.print(f"❌ Error extracting segments for {speaker_name}: {e}", style="red")
            return AudioSegment.empty()
    
    def create_speaker_files_from_timestamps(self, audio_file: str, timestamp_data: Dict[str, List[Tuple[str, str]]], output_dir: str = None) -> List[str]:
        """Create separate audio files for each speaker based on timestamps"""
        output_files = []
        
        # Use current directory if no output dir specified
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        console.print(f"\n🎭 [bold]Creating speaker files from timestamps...[/bold]", style="blue")
        console.print(f"📁 Source audio: {os.path.basename(audio_file)}")
        console.print(f"📂 Output directory: {output_dir}")
        console.print(f"👥 Speakers: {list(timestamp_data.keys())}")
        
        for speaker_name, timestamps in timestamp_data.items():
            console.print(f"\n🎤 Processing [bold]{speaker_name}[/bold] ({len(timestamps)} segments)")
            
            # Extract and stitch segments for this speaker
            speaker_audio = self.extract_audio_segments_from_timestamps(audio_file, timestamps, speaker_name)
            
            if len(speaker_audio) > 0:
                # Create safe filename
                safe_speaker_name = speaker_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                output_path = os.path.join(output_dir, f"speaker_{safe_speaker_name}.wav")
                speaker_audio.export(output_path, format="wav")
                output_files.append(output_path)
                console.print(f"💾 Saved: {output_path}", style="green")
            else:
                console.print(f"⚠️ No audio extracted for {speaker_name}", style="yellow")
        
        console.print(f"\n✅ Created {len(output_files)} speaker files in {output_dir}", style="green")
        return output_files
    
    def remove_audio_segments_by_timestamps(self, audio_file: str, removal_timestamps: List[Tuple[str, str]], output_path: str = None) -> str:
        """Remove specific audio segments based on timestamps"""
        try:
            # Load the source audio file
            audio = AudioSegment.from_file(audio_file)
            console.print(f"🎵 Loaded audio: {len(audio)/1000:.1f}s", style="cyan")
            
            # Convert timestamps to milliseconds and sort them
            removal_intervals = []
            for start_str, end_str in removal_timestamps:
                start_seconds = self.parse_timestamp(start_str)
                end_seconds = self.parse_timestamp(end_str)
                start_ms = int(start_seconds * 1000)
                end_ms = int(end_seconds * 1000)
                
                # Validate interval
                if start_ms < len(audio) and end_ms <= len(audio) and start_ms < end_ms:
                    removal_intervals.append((start_ms, end_ms))
                    console.print(f"  🗑️ Will remove: {start_str} - {end_str} ({(end_ms-start_ms)/1000:.1f}s)", style="yellow")
                else:
                    console.print(f"  ⚠️ Invalid removal range: {start_str} - {end_str}", style="yellow")
            
            if not removal_intervals:
                console.print("❌ No valid removal intervals found", style="red")
                return audio_file
            
            # Sort intervals by start time
            removal_intervals.sort(key=lambda x: x[0])
            
            console.print(f"🔪 Removing {len(removal_intervals)} segments...", style="cyan")
            
            # Build new audio by keeping segments between removals
            result_audio = AudioSegment.empty()
            last_end = 0
            total_removed_duration = 0
            
            for start_ms, end_ms in removal_intervals:
                # Add the segment before this removal
                if start_ms > last_end:
                    segment = audio[last_end:start_ms]
                    result_audio += segment
                    console.print(f"  ✅ Kept: {last_end/1000:.1f}s - {start_ms/1000:.1f}s ({len(segment)/1000:.1f}s)", style="green")
                
                # Track removed duration
                removed_duration = (end_ms - start_ms) / 1000
                total_removed_duration += removed_duration
                console.print(f"  🗑️ Removed: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s ({removed_duration:.1f}s)", style="red")
                
                last_end = end_ms
            
            # Add the remaining audio after the last removal
            if last_end < len(audio):
                final_segment = audio[last_end:]
                result_audio += final_segment
                console.print(f"  ✅ Kept final: {last_end/1000:.1f}s - {len(audio)/1000:.1f}s ({len(final_segment)/1000:.1f}s)", style="green")
            
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                output_dir = os.path.dirname(audio_file) or os.getcwd()
                output_path = os.path.join(output_dir, f"{base_name}_cleaned.wav")
            
            # Save the cleaned audio
            result_audio.export(output_path, format="wav")
            
            original_duration = len(audio) / 1000
            final_duration = len(result_audio) / 1000
            
            console.print(f"✅ [bold green]Audio cleaned successfully![/bold green]", style="green")
            console.print(f"📊 Original: {original_duration:.1f}s → Final: {final_duration:.1f}s (removed {total_removed_duration:.1f}s)", style="cyan")
            console.print(f"💾 Saved: {output_path}", style="green")
            
            return output_path
            
        except Exception as e:
            console.print(f"❌ Error removing audio segments: {e}", style="red")
            return audio_file
    
    def download_all_sources(self, input_sources: List[str]) -> List[str]:
        """Download all audio sources first"""
        audio_files = []
        
        console.print(f"\n📥 [bold]Downloading {len(input_sources)} audio sources...[/bold]", style="blue")
        
        for i, input_source in enumerate(input_sources):
            console.print(f"\n📹 [bold]Source {i + 1}/{len(input_sources)}[/bold]: [blue]{input_source}[/blue]")
            
            if input_source.startswith(('http://', 'https://')):
                # Download YouTube video
                audio_path = self.download_youtube_audio(input_source, i)
            else:
                # Copy local file to temp directory
                filename = f"input_audio_{i}.wav"
                audio_path = os.path.join(self.temp_dir, filename)
                shutil.copy2(input_source, audio_path)
                console.print("✅ Audio file loaded", style="green")
            
            audio_files.append(audio_path)
        
        return audio_files
    
    def stitch_all_audio(self, audio_files: List[str]) -> str:
        """Stitch all audio files together with silence buffers and VAD pre-processing"""
        console.print(f"\n🔗 [bold]Stitching {len(audio_files)} audio files together...[/bold]", style="cyan")
        
        combined_audio = AudioSegment.empty()
        silence_buffer = AudioSegment.silent(duration=500)  # 0.5 second buffer between segments
        
        total_original_duration = 0
        total_processed_duration = 0
        
        for i, audio_file in enumerate(audio_files):
            try:
                audio = AudioSegment.from_file(audio_file)
                original_duration = len(audio) / 1000
                total_original_duration += original_duration
                
                console.print(f"  📎 Video {i+1}: {original_duration:.1f}s → processing...", style="dim")
                
                # Apply VAD to remove silence/gaps before stitching
                cleaned_audio = self.remove_silence_with_vad(audio)
                processed_duration = len(cleaned_audio) / 1000
                total_processed_duration += processed_duration
                
                console.print(f"    🔇 After VAD: {processed_duration:.1f}s", style="dim cyan")
                
                # Add the cleaned audio
                combined_audio += cleaned_audio
                
                # Add buffer between videos (except after last one)
                if i < len(audio_files) - 1:
                    combined_audio += silence_buffer
                    
            except Exception as e:
                console.print(f"⚠️ Could not load {audio_file}: {e}", style="yellow")
        
        # Save combined audio
        combined_path = os.path.join(self.temp_dir, "combined_audio.wav")
        combined_audio.export(combined_path, format="wav")
        
        console.print(f"✅ Combined audio: {total_original_duration:.1f}s → {total_processed_duration:.1f}s from {len(audio_files)} sources", style="green")
        return combined_path
    
    def process(self, input_sources: List[str]) -> List[str]:
        """Main processing function for multiple sources"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup pipelines
            self.setup_pipeline()
            self.setup_music_separator()
            
            # Step 1: Download all audio sources
            audio_files = self.download_all_sources(input_sources)
            
            if not audio_files:
                console.print("❌ No audio files downloaded", style="red")
                return []
            
            # Step 2: Stitch all audio together
            combined_audio_path = self.stitch_all_audio(audio_files)
            
            # Step 3: Remove background music from combined audio (if enabled)
            clean_audio_path = self.remove_background_music(combined_audio_path)
            
            # Step 4: Separate voices from the combined audio
            console.print(f"\n🗣️ [bold]Running voice separation on combined audio...[/bold]", style="blue")
            voice_files = self.separate_voices(clean_audio_path)
            
            if not voice_files:
                console.print("❌ No voice tracks found in combined audio", style="red")
                return []
            
            console.print(f"✅ Found {len(voice_files)} voice tracks from combined audio", style="green")
            
            # Step 5: Interactive voice selection and labeling
            selected_voices = self.interactive_voice_selection(voice_files)
            
            if not selected_voices:
                console.print("❌ No voices were selected", style="red")
                return []
            
            # Step 6: Label the voices (now they're already long tracks from multiple videos)
            labeled_voices = self.interactive_labeling(selected_voices, 0)
            
            if not labeled_voices:
                console.print("❌ No voices were labeled", style="red")
                return []
            
            # Step 7: Final processing and model creation
            console.print(f"\n🎭 [bold]Creating Models for {len(labeled_voices)} Characters[/bold]", style="blue")
            
            model_ids = []
            for voice_file, character_name, description in labeled_voices:
                # Load and process the voice
                audio = AudioSegment.from_file(voice_file)
                
                # Remove silence from the voice track
                cleaned_audio = self.remove_silence_with_vad(audio)
                
                if len(cleaned_audio) < 1000:  # Less than 1 second
                    console.print(f"⚠️ Not enough audio for {character_name} ({len(cleaned_audio)/1000:.1f}s)", style="yellow")
                    continue
                
                # Save final processed audio
                final_audio_path = os.path.join(self.temp_dir, f"final_{character_name}.wav")
                cleaned_audio.export(final_audio_path, format="wav")
                
                # Create Fish Audio model
                model_id = self.create_fish_model(final_audio_path, character_name, description)
                if model_id:
                    model_ids.append(model_id)
            
            return model_ids
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def process_with_gemini(self, input_sources: List[str]) -> List[str]:
        """Main processing function using Gemini for speaker separation"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup music separator only
            self.setup_music_separator()
            
            if not self.use_gemini:
                console.print("❌ Gemini is not enabled. Use --gemini-api-key to enable.", style="red")
                return []
            
            # Step 1: Download all audio sources
            audio_files = self.download_all_sources(input_sources)
            
            if not audio_files:
                console.print("❌ No audio files downloaded", style="red")
                return []
            
            # Step 2: Stitch all audio together
            combined_audio_path = self.stitch_all_audio(audio_files)
            
            # Step 3: Remove background music from combined audio (if enabled)
            clean_audio_path = self.remove_background_music(combined_audio_path)
            
            # Step 4: Analyze audio with Gemini to get speaker timestamps
            console.print(f"\n🤖 [bold]Analyzing combined audio with Gemini...[/bold]", style="blue")
            speaker_timestamps = self.analyze_audio_with_gemini(clean_audio_path)
            
            if not speaker_timestamps:
                console.print("❌ Gemini could not identify any speakers", style="red")
                return []
            
            console.print(f"✅ Gemini found {len(speaker_timestamps)} speakers", style="green")
            
            # Step 5: Create initial speaker files from Gemini timestamps
            output_dir = os.path.join(self.temp_dir, "gemini_speakers")
            initial_speaker_files = self.create_speaker_files_from_timestamps(clean_audio_path, speaker_timestamps, output_dir)
            
            if not initial_speaker_files:
                console.print("❌ No speaker files created from Gemini analysis", style="red")
                return []
            
            # Step 6: Interactive review and refinement
            console.print(f"\n🎭 [bold]Review and refine speakers...[/bold]", style="blue")
            model_ids = []
            
            for speaker_file in initial_speaker_files:
                # Get speaker name from filename
                filename = os.path.basename(speaker_file)
                # Extract original speaker name from filename (remove "speaker_" prefix and ".wav" suffix)
                original_speaker_name = filename.replace("speaker_", "").replace(".wav", "").replace("_", " ")
                
                console.print(f"\n🎤 [bold]Processing: {original_speaker_name}[/bold]", style="cyan")
                
                # Ask user if they want to keep this speaker
                if not Confirm.ask(f"Keep {original_speaker_name}?", default=True):
                    console.print("⏭️ Skipping speaker", style="yellow")
                    continue
                
                # Let user edit the character name and description
                character_name = Prompt.ask("Character name", default=original_speaker_name.split("(")[0].strip())
                description = Prompt.ask("Character description", default=original_speaker_name)
                
                # Refine with Gemini for high-confidence segments
                console.print(f"🤖 Asking Gemini to refine timestamps for [bold]{character_name}[/bold]...", style="cyan")
                refined_timestamps = self.refine_speaker_with_gemini(speaker_file, character_name, description)
                
                if not refined_timestamps:
                    console.print(f"⚠️ Gemini couldn't find confident segments for {character_name}", style="yellow")
                    
                    # Ask if user wants to keep the original version
                    if Confirm.ask("Keep original version anyway?", default=False):
                        refined_timestamps = speaker_timestamps.get(original_speaker_name, [])
                    else:
                        continue
                
                # Create final refined audio
                if refined_timestamps:
                    console.print(f"🔧 Creating refined audio for [bold]{character_name}[/bold]...", style="cyan")
                    final_audio = self.extract_audio_segments_from_timestamps(speaker_file, refined_timestamps, character_name)
                    
                    if len(final_audio) > 5000:  # At least 5 seconds
                        # Save final audio
                        final_audio_path = os.path.join(self.temp_dir, f"final_{character_name.replace(' ', '_')}.wav")
                        final_audio.export(final_audio_path, format="wav")
                        
                        # Create Fish Audio model
                        model_id = self.create_fish_model(final_audio_path, character_name, description)
                        if model_id:
                            model_ids.append(model_id)
                            console.print(f"✅ Created model for [bold]{character_name}[/bold]: {model_id}", style="green")
                    else:
                        console.print(f"⚠️ Not enough refined audio for {character_name} ({len(final_audio)/1000:.1f}s)", style="yellow")
            
            return model_ids
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)


def test_timestamp_extraction():
    """Test function with your sample timestamp data"""
    # Sample timestamp data
    timestamp_data = {
        "Speaker 1 (Deeper, raspy voice)": [
            ("0:06", "0:08"), ("0:22", "0:25"), ("0:28", "0:39"), ("1:15", "1:18"),
            ("1:21", "1:22"), ("1:23", "1:28"), ("1:46", "1:51"), ("2:03", "2:04"),
            ("2:51", "2:57"), ("2:58", "2:59"), ("3:42", "3:44"), ("4:08", "4:11"),
            ("4:14", "4:15"), ("4:18", "4:21"), ("4:30", "4:33"), ("4:42", "4:44"),
            ("4:56", "5:02")
        ],
        "Speaker 2 (Higher-pitched, more excitable voice)": [
            ("0:03", "0:05"), ("0:09", "0:09"), ("0:13", "0:16"), ("0:49", "0:50"),
            ("1:05", "1:07"), ("1:12", "1:14"), ("1:32", "1:35"), ("1:37", "1:43"),
            ("1:52", "2:02"), ("2:21", "2:22"), ("2:27", "2:28"), ("3:24", "3:25"),
            ("3:48", "3:52"), ("4:53", "4:55"), ("6:13", "6:20"), ("6:35", "6:36"),
            ("6:46", "6:49"), ("6:51", "6:53"), ("7:04", "7:11"), ("7:16", "7:21"),
            ("7:26", "7:32"), ("8:01", "8:08"), ("8:11", "8:18"), ("8:22", "8:29"),
            ("8:50", "8:51"), ("8:54", "8:56"), ("9:06", "9:08"), ("9:19", "9:21"),
            ("9:25", "9:27")
        ],
        "Speaker 3 (Whiny, nasally voice)": [
            ("0:18", "0:21"), ("0:54", "0:57"), ("1:18", "1:20"), ("1:22", "1:23"),
            ("1:29", "1:30"), ("1:35", "1:37"), ("1:44", "1:45"), ("2:28", "2:29"),
            ("3:26", "3:28"), ("3:44", "3:48"), ("4:09", "4:10"), ("4:15", "4:16"),
            ("4:24", "4:26"), ("5:03", "5:05"), ("5:07", "5:14"), ("5:25", "5:31"),
            ("6:01", "6:09"), ("6:20", "6:30"), ("6:32", "6:34"), ("6:37", "6:40"),
            ("7:12", "7:16"), ("7:21", "7:26"), ("7:28", "7:31"), ("8:30", "8:31"),
            ("8:39", "8:45"), ("8:46", "8:49"), ("9:21", "9:24"), ("9:27", "9:28")
        ],
        "Speaker 4 (Calm, normal-pitched male voice)": [
            ("0:16", "0:18"), ("0:25", "0:28"), ("0:40", "0:45"), ("0:48", "0:49"),
            ("0:51", "0:51"), ("1:01", "1:04"), ("1:07", "1:08"), ("3:30", "3:38")
        ]
    }
    
    console.print("\n🧪 [bold]Running Timestamp Extraction Test[/bold]", style="blue")
    console.print("This will extract audio segments based on the provided timestamps")
    
    # Get audio file path from user
    audio_file = Prompt.ask("📁 Enter path to audio file (vocal/separated audio)")
    
    if not os.path.exists(audio_file):
        console.print(f"❌ Audio file not found: {audio_file}", style="red")
        return
    
    # Get output directory from user
    output_dir = Prompt.ask("📂 Output directory for speaker files", default="./speaker_outputs")
    
    # Create trainer instance (no tokens needed for timestamp extraction)
    trainer = VoiceTrainer("dummy", "dummy")
    
    # Extract speaker files to permanent location
    output_files = trainer.create_speaker_files_from_timestamps(audio_file, timestamp_data, output_dir)
    
    if output_files:
        console.print(f"\n🎉 [bold green]Success![/bold green] Created {len(output_files)} speaker files:", style="green")
        for file_path in output_files:
            console.print(f"  📄 {file_path}", style="cyan")
            
            # Show file info
            try:
                audio = AudioSegment.from_file(file_path)
                console.print(f"     Duration: {len(audio)/1000:.1f}s", style="dim")
            except Exception as e:
                console.print(f"     Error loading: {e}", style="yellow")
                
        console.print(f"\n💡 [bold]Files saved permanently in:[/bold] {os.path.abspath(output_dir)}", style="blue")
    else:
        console.print("❌ No speaker files created", style="red")


def test_sound_removal():
    """Test function for sound removal with sample timestamps"""
    # Sample removal timestamps
    removal_timestamps = [
       ("00:05", "00:06"),
       ("00:17", "00:20"),
       ("00:22", "00:23"),
       ("00:33", "00:34"),
       ("00:41", "00:42"),
       ("00:43", "00:45"),
       ("00:45", "00:47"),
       ("00:47", "00:48"),
       ("00:49", "00:50"),
       ("00:51", "00:53"),
       ("00:54", "00:56"),
       ("00:57", "01:02"),
    ]
    
    console.print("\n🧪 [bold]Running Sound Removal Test[/bold]", style="blue")
    console.print("This will remove specific audio segments based on timestamps")
    
    # Get audio file path from user
    audio_file = Prompt.ask("📁 Enter path to audio file")
    
    if not os.path.exists(audio_file):
        console.print(f"❌ Audio file not found: {audio_file}", style="red")
        return
    
    # Get output file path
    default_output = os.path.splitext(audio_file)[0] + "_cleaned.wav"
    output_file = Prompt.ask("💾 Output file path", default=default_output)
    
    # Create trainer instance (no tokens needed for audio processing)
    trainer = VoiceTrainer("dummy", "dummy")
    
    # Remove audio segments
    console.print(f"\n🗑️ [bold]Removing {len(removal_timestamps)} audio segments...[/bold]", style="yellow")
    
    result_file = trainer.remove_audio_segments_by_timestamps(audio_file, removal_timestamps, output_file)
    
    if result_file != audio_file:
        console.print(f"\n🎉 [bold green]Success![/bold green] Audio cleaned and saved:", style="green")
        console.print(f"  📄 {result_file}", style="cyan")
        
        # Show file info
        try:
            original_audio = AudioSegment.from_file(audio_file)
            cleaned_audio = AudioSegment.from_file(result_file)
            original_duration = len(original_audio) / 1000
            cleaned_duration = len(cleaned_audio) / 1000
            removed_duration = original_duration - cleaned_duration
            
            console.print(f"📊 Summary:", style="bold")
            console.print(f"  • Original: {original_duration:.1f}s", style="dim")
            console.print(f"  • Cleaned: {cleaned_duration:.1f}s", style="dim")
            console.print(f"  • Removed: {removed_duration:.1f}s", style="dim")
            console.print(f"  • Reduction: {(removed_duration/original_duration)*100:.1f}%", style="dim")
        except Exception as e:
            console.print(f"     Error analyzing files: {e}", style="yellow")
    else:
        console.print("❌ No audio processing completed", style="red")


@click.group()
def cli():
    """Super Voice Auto Trainer - Multi-Video Character Voice Training"""
    pass


@cli.command(name="test-timestamps")
def test_timestamps_cmd():
    """Test timestamp extraction with sample data"""
    test_timestamp_extraction()


@cli.command(name="test-removal")
def test_removal_cmd():
    """Test sound removal with sample timestamps"""
    test_sound_removal()


@cli.command(name="train")
@click.argument('input_sources', nargs=-1, required=True)
@click.option('--hf-token', envvar='HF_TOKEN', required=False, 
              help='Hugging Face access token (or set HF_TOKEN env var) - Required for pyannote')
@click.option('--fish-api-key', envvar='FISH_API_KEY', required=True,
              help='Fish Audio API key (or set FISH_API_KEY env var)')
@click.option('--gemini-api-key', envvar='GEMINI_API_KEY', required=False,
              help='Gemini API key (or set GEMINI_API_KEY env var) - Use for Gemini-based speaker separation')
@click.option('--use-gemini', is_flag=True, default=False,
              help='Use Gemini for speaker separation instead of pyannote')
@click.option('--remove-music', is_flag=True, default=False,
              help='Remove background music before voice separation')
@click.option('--separator-model', default='UVR-MDX-NET-Inst_HQ_3.onnx',
                help='Model to use for music separation. Defaults to UVR-MDX-NET-Inst_HQ_3.onnx')
@click.option('--skip-music-separation', is_flag=True, default=False,
              help='Skip music separation step (useful if already processed)')
def main(input_sources: tuple, hf_token: str, fish_api_key: str, gemini_api_key: str, use_gemini: bool, remove_music: bool, separator_model: str, skip_music_separation: bool):
    """
    Super Voice Auto Trainer - Multi-Video Character Voice Training
    
    INPUT_SOURCES can be multiple:
    - YouTube URLs (https://www.youtube.com/watch?v=...)
    - Local audio file paths (.wav, .mp3, .mp4, etc.)
    
    Example:
    voice-trainer "https://youtube.com/watch?v=vid1" "https://youtube.com/watch?v=vid2" "local_file.wav"
    
    The tool will process all sources, allow you to match characters across videos,
    and create combined voice models with stitched audio from multiple sources.
    """
    # Header with enhanced styling
    console.print(Panel.fit(
        "🎤 [bold blue]Super Voice Auto Trainer[/bold blue] 🎤\n"
        "Multi-Video Character Voice Training System",
        style="cyan"
    ))
    
    # Show input sources
    sources_table = Table(title="Input Sources")
    sources_table.add_column("#", style="cyan", no_wrap=True)
    sources_table.add_column("Source", style="green")
    sources_table.add_column("Type", style="yellow")
    
    for i, source in enumerate(input_sources, 1):
        source_type = "YouTube URL" if source.startswith(('http://', 'https://')) else "Local File"
        sources_table.add_row(str(i), source, source_type)
    
    console.print(sources_table)
    
    # Show configuration
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    music_status = "⏭️ Skipped" if skip_music_separation else ("✅ Enabled" if remove_music else "❌ Disabled")
    config_table.add_row("Music Removal", music_status)
    if remove_music and not skip_music_separation:
        config_table.add_row("Separator Model", separator_model)
    config_table.add_row("GPU Support", "✅ Available" if torch.cuda.is_available() else "❌ CPU Only")
    config_table.add_row("Multi-Video Mode", "✅ Enabled")
    config_table.add_row("Voice Activity Detection", "✅ Enabled")
    config_table.add_row("VAD Pre-processing", "✅ Enabled (before voice separation)")
    
    # Determine processing method
    if use_gemini:
        if not gemini_api_key:
            console.print("❌ Gemini API key required when using --use-gemini", style="red")
            return
        config_table.add_row("Speaker Separation", "🤖 Gemini AI")
        config_table.add_row("Refinement", "🤖 Gemini AI (High Confidence)")
    else:
        if not hf_token:
            console.print("❌ Hugging Face token required for pyannote speech separation", style="red")
            return
        config_table.add_row("Speaker Separation", "🔬 pyannote")
    
    console.print(config_table)
    
    # Create trainer with appropriate settings
    trainer = VoiceTrainer(
        hf_token=hf_token or "dummy", 
        fish_api_key=fish_api_key, 
        remove_music=remove_music, 
        separator_model=separator_model, 
        skip_music_separation=skip_music_separation,
        gemini_api_key=gemini_api_key,
        use_gemini=use_gemini
    )
    
    # Choose processing method
    if use_gemini:
        console.print("\n🤖 [bold blue]Using Gemini AI for speaker separation[/bold blue]", style="cyan")
        model_ids = trainer.process_with_gemini(list(input_sources))
    else:
        console.print("\n🔬 [bold blue]Using pyannote for speaker separation[/bold blue]", style="cyan")
        model_ids = trainer.process(list(input_sources))
    
    # Results summary
    if model_ids:
        console.print(Panel.fit(
            f"🎉 [bold green]Success![/bold green]\n"
            f"Created {len(model_ids)} character voice models from {len(input_sources)} sources",
            style="green"
        ))
        
        # Model IDs table
        results_table = Table(title="Created Character Models")
        results_table.add_column("#", style="cyan", no_wrap=True)
        results_table.add_column("Character Model ID", style="green")
        
        for i, model_id in enumerate(model_ids, 1):
            results_table.add_row(str(i), model_id)
        
        console.print(results_table)
        
        console.print("\n🎯 [bold]Pro Tip:[/bold] These models are trained on stitched audio from multiple sources with silence removed!")
    else:
        console.print(Panel.fit(
            "❌ [bold red]No models were created[/bold red]\n"
            "Check the logs above for details",
            style="red"
        ))


if __name__ == "__main__":
    cli()