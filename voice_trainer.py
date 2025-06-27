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
from typing import List, Tuple, Optional
import requests
import click
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch


class VoiceTrainer:
    def __init__(self, hf_token: str, fish_api_key: str):
        self.hf_token = hf_token
        self.fish_api_key = fish_api_key
        self.pipeline = None
        self.temp_dir = None
        
    def setup_pipeline(self):
        """Initialize the pyannote pipeline"""
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speech-separation-ami-1.0",
                use_auth_token=self.hf_token
            )
            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                click.echo("Using GPU for processing")
            else:
                click.echo("Using CPU for processing")
        except Exception as e:
            click.echo(f"Error setting up pyannote pipeline: {e}")
            sys.exit(1)
    
    def download_youtube_audio(self, url: str) -> str:
        """Download audio from YouTube URL using yt-dlp"""
        output_path = os.path.join(self.temp_dir, "downloaded_audio.%(ext)s")
        
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "-o", output_path,
            url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Find the actual output file
            for file in os.listdir(self.temp_dir):
                if file.startswith("downloaded_audio") and file.endswith(".wav"):
                    return os.path.join(self.temp_dir, file)
            raise FileNotFoundError("Downloaded audio file not found")
        except subprocess.CalledProcessError as e:
            click.echo(f"Error downloading YouTube video: {e.stderr}")
            sys.exit(1)
    
    def separate_voices(self, audio_path: str) -> List[str]:
        """Separate voices using pyannote speech separation"""
        click.echo("Separating voices...")
        
        try:
            diarization, sources = self.pipeline(audio_path)
            
            separated_files = []
            for i, (_, source) in enumerate(sources.items()):
                # Convert to AudioSegment for easier manipulation
                temp_path = os.path.join(self.temp_dir, f"temp_source_{i}.wav")
                source_audio = AudioSegment.from_file(temp_path)
                
                # Trim to 45-60 seconds
                duration_ms = len(source_audio)
                if duration_ms > 60000:  # More than 60 seconds
                    trimmed_audio = source_audio[:55000]  # Take first 55 seconds
                elif duration_ms < 45000:  # Less than 45 seconds
                    # Skip if too short
                    continue
                else:
                    trimmed_audio = source_audio
                
                # Save trimmed audio
                output_path = os.path.join(self.temp_dir, f"voice_{i}.wav")
                trimmed_audio.export(output_path, format="wav")
                separated_files.append(output_path)
            
            return separated_files
            
        except Exception as e:
            click.echo(f"Error separating voices: {e}")
            return []
    
    def interactive_labeling(self, voice_files: List[str]) -> List[Tuple[str, str, str]]:
        """Interactive labeling of voice tracks"""
        labeled_voices = []
        
        click.echo(f"\nFound {len(voice_files)} voice tracks to label:")
        
        for i, voice_file in enumerate(voice_files):
            click.echo(f"\n--- Voice Track {i+1} ---")
            click.echo(f"File: {voice_file}")
            
            # Play audio (if possible)
            try:
                if shutil.which("play"):  # Sox play command
                    subprocess.run(["play", voice_file], check=True, capture_output=True)
                elif shutil.which("aplay"):  # ALSA player
                    subprocess.run(["aplay", voice_file], check=True, capture_output=True)
                else:
                    click.echo("(Audio playback not available - install sox or alsa-utils)")
            except:
                click.echo("(Could not play audio)")
            
            # Get user input
            name = click.prompt("Enter character/speaker name (or 'skip' to skip this track)")
            
            if name.lower() == 'skip':
                continue
                
            description = click.prompt("Enter description for this voice")
            
            labeled_voices.append((voice_file, name, description))
        
        return labeled_voices
    
    def create_fish_model(self, audio_file: str, name: str, description: str) -> Optional[str]:
        """Create a model on Fish Audio platform"""
        url = "https://api.fish.audio/model"
        
        headers = {
            "Authorization": f"Bearer {self.fish_api_key}"
        }
        
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
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                model_id = result.get('id')
                
                if model_id:
                    click.echo(f"✓ Created model '{name}' with ID: {model_id}")
                    return model_id
                else:
                    click.echo(f"✗ Failed to get model ID for '{name}'")
                    return None
                    
            except requests.exceptions.RequestException as e:
                click.echo(f"✗ Error creating model '{name}': {e}")
                return None
    
    def process(self, input_source: str) -> List[str]:
        """Main processing function"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup pipeline
            self.setup_pipeline()
            
            # Download or copy audio
            if input_source.startswith(('http://', 'https://')):
                click.echo(f"Downloading audio from: {input_source}")
                audio_path = self.download_youtube_audio(input_source)
            else:
                # Copy local file to temp directory
                audio_path = os.path.join(self.temp_dir, "input_audio.wav")
                shutil.copy2(input_source, audio_path)
            
            # Separate voices
            voice_files = self.separate_voices(audio_path)
            
            if not voice_files:
                click.echo("No voice tracks found or separated")
                return []
            
            # Interactive labeling
            labeled_voices = self.interactive_labeling(voice_files)
            
            if not labeled_voices:
                click.echo("No voices were labeled")
                return []
            
            # Create models on Fish Audio
            model_ids = []
            click.echo("\nCreating models on Fish Audio...")
            
            for audio_file, name, description in labeled_voices:
                model_id = self.create_fish_model(audio_file, name, description)
                if model_id:
                    model_ids.append(model_id)
            
            return model_ids
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)


@click.command()
@click.argument('input_source')
@click.option('--hf-token', envvar='HF_TOKEN', required=True, 
              help='Hugging Face access token (or set HF_TOKEN env var)')
@click.option('--fish-api-key', envvar='FISH_API_KEY', required=True,
              help='Fish Audio API key (or set FISH_API_KEY env var)')
def main(input_source: str, hf_token: str, fish_api_key: str):
    """
    Super Voice Auto Trainer
    
    INPUT_SOURCE can be either:
    - A YouTube URL (https://www.youtube.com/watch?v=...)
    - A local audio file path (.wav, .mp3, .mp4, etc.)
    """
    click.echo("🎤 Super Voice Auto Trainer")
    click.echo("=" * 40)
    
    trainer = VoiceTrainer(hf_token, fish_api_key)
    model_ids = trainer.process(input_source)
    
    if model_ids:
        click.echo(f"\n🎉 Successfully created {len(model_ids)} voice models!")
        click.echo("Model IDs:")
        for i, model_id in enumerate(model_ids, 1):
            click.echo(f"  {i}. {model_id}")
    else:
        click.echo("\n❌ No models were created")


if __name__ == "__main__":
    main()