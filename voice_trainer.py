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

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# Load environment variables from .env file
load_dotenv()

console = Console()


class VoiceTrainer:
    def __init__(self, hf_token: str, fish_api_key: str, remove_music: bool = False, separator_model: Optional[str] = None, skip_music_separation: bool = False):
        self.hf_token = hf_token
        self.fish_api_key = fish_api_key
        self.remove_music = remove_music
        self.separator_model = separator_model
        self.skip_music_separation = skip_music_separation
        self.pipeline = None
        self.separator = None
        self.temp_dir = None
        self.downloads_dir = os.path.join(os.getcwd(), "downloads")  # Persistent downloads
        self.character_voices = defaultdict(list)  # Store voice clips by character name
        
        # Create persistent downloads directory
        os.makedirs(self.downloads_dir, exist_ok=True)
        
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
        console.print("🗣️ Separating voices...", style="cyan")
        
        try:
            # Use pyannote's built-in progress tracking
            with ProgressHook() as hook:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Analyzing and separating speakers...", total=None)
                    diarization, sources = self.pipeline(audio_path, hook=hook)
                    progress.update(task, completed=100)
            
            separated_files = []
            for i, (_, source) in enumerate(sources.items()):
                # Convert to AudioSegment for easier manipulation
                temp_path = os.path.join(self.temp_dir, f"temp_source_{i}.wav")
                source_audio = AudioSegment.from_file(temp_path)
                
                # Save the full audio without trimming
                output_path = os.path.join(self.temp_dir, f"voice_{i}.wav")
                source_audio.export(output_path, format="wav")
                separated_files.append(output_path)
            
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


@click.command()
@click.argument('input_sources', nargs=-1, required=True)
@click.option('--hf-token', envvar='HF_TOKEN', required=True, 
              help='Hugging Face access token (or set HF_TOKEN env var)')
@click.option('--fish-api-key', envvar='FISH_API_KEY', required=True,
              help='Fish Audio API key (or set FISH_API_KEY env var)')
@click.option('--remove-music', is_flag=True, default=False,
              help='Remove background music before voice separation')
@click.option('--separator-model', default='UVR-MDX-NET-Inst_HQ_3.onnx',
                help='Model to use for music separation. Defaults to UVR-MDX-NET-Inst_HQ_3.onnx')
@click.option('--skip-music-separation', is_flag=True, default=False,
              help='Skip music separation step (useful if already processed)')
def main(input_sources: tuple, hf_token: str, fish_api_key: str, remove_music: bool, separator_model: str, skip_music_separation: bool):
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
    console.print(config_table)
    
    print(hf_token)
    print(fish_api_key)
    
    trainer = VoiceTrainer(hf_token, fish_api_key, remove_music, separator_model, skip_music_separation)
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
    main()