"""
Core VoiceTrainer class for voice model training
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
import requests
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

from ..audio.processing import AudioProcessor
from ..gemini.analyzer import GeminiAnalyzer
from ..utils.helpers import create_safe_filename

console = Console()


class VoiceTrainer:
    """Main voice trainer class"""
    
    def __init__(self, hf_token: str, fish_api_key: str, remove_music: bool = False, 
                 separator_model: Optional[str] = None, skip_music_separation: bool = False, 
                 gemini_api_key: Optional[str] = None, use_gemini: bool = False):
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
        self.downloads_dir = os.path.join(os.getcwd(), "downloads")
        self.character_voices = defaultdict(list)
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        if self.use_gemini and self.gemini_api_key:
            self.gemini_analyzer = GeminiAnalyzer(self.gemini_api_key)
        else:
            self.gemini_analyzer = None
        
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
            self.separator = Separator()
            if self.separator_model:
                console.print(f"🎵 Loading separator model: [bold]{self.separator_model}[/bold]...", style="cyan")
                self.separator.load_model(self.separator_model)
            console.print("✅ Music separator ready", style="green")
        except Exception as e:
            console.print(f"❌ Error setting up music separator: {e}", style="red")
            console.print("⚠️ Music separation will be disabled", style="yellow")
            self.separator = None
    
    def create_fish_model(self, audio_file: str, character_name: str, description: str) -> Optional[str]:
        """Create a Fish Audio model"""
        try:
            console.print(f"🐟 Creating Fish Audio model for [bold]{character_name}[/bold]...", style="cyan")
            
            # Fish Audio API endpoint (placeholder - you'll need the actual endpoint)
            url = "https://api.fish.audio/v1/models"
            
            headers = {
                "Authorization": f"Bearer {self.fish_api_key}",
                "Content-Type": "application/json"
            }
            
            # Load audio file
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            data = {
                "name": character_name,
                "description": description,
                "audio": audio_data
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                model_data = response.json()
                model_id = model_data.get("model_id")
                console.print(f"✅ Created model: {model_id}", style="green")
                return model_id
            else:
                console.print(f"❌ Failed to create model: {response.text}", style="red")
                return None
                
        except Exception as e:
            console.print(f"❌ Error creating Fish model: {e}", style="red")
            return None
    
    def process(self, input_sources: List[str]) -> List[str]:
        """Main processing function for multiple sources using pyannote"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup pipelines
            self.setup_pipeline()
            self.setup_music_separator()
            
            # Step 1: Download all audio sources
            audio_files = self.audio_processor.download_all_sources(
                input_sources, self.downloads_dir, self.temp_dir
            )
            
            if not audio_files:
                console.print("❌ No audio files downloaded", style="red")
                return []
            
            # Step 2: Process audio (combine, clean, separate voices)
            model_ids = self.audio_processor.process_audio_files(
                audio_files, self.temp_dir, self.remove_music, 
                self.separator, self.skip_music_separation, self.pipeline
            )
            
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
            
            if not self.gemini_analyzer:
                console.print("❌ Gemini is not enabled. Use --gemini-api-key to enable.", style="red")
                return []
            
            # Step 1: Download and process audio sources
            audio_files = self.audio_processor.download_all_sources(
                input_sources, self.downloads_dir, self.temp_dir
            )
            
            if not audio_files:
                console.print("❌ No audio files downloaded", style="red")
                return []
            
            # Step 2: Combine and clean audio
            combined_audio_path = self.audio_processor.stitch_all_audio(audio_files, self.temp_dir)
            clean_audio_path = self.audio_processor.remove_background_music(
                combined_audio_path, self.separator, self.remove_music, self.skip_music_separation
            )
            
            # Step 3: Ask user for target speakers (optional)
            target_speakers = self._get_target_speakers()
            
            # Step 4: Analyze with Gemini
            console.print(f"\n🤖 [bold]Analyzing combined audio with Gemini...[/bold]", style="blue")
            if target_speakers:
                console.print(f"🎯 Targeting specific speakers: {', '.join(target_speakers)}", style="cyan")
            speaker_timestamps = self.gemini_analyzer.analyze_audio(clean_audio_path, target_speakers)
            
            if not speaker_timestamps:
                console.print("❌ Gemini could not identify any speakers", style="red")
                return []
            
            console.print(f"✅ Gemini found {len(speaker_timestamps)} speakers", style="green")
            
            # Step 4: Create initial speaker files
            output_dir = os.path.join(self.temp_dir, "gemini_speakers")
            initial_speaker_files = self.audio_processor.create_speaker_files_from_timestamps(
                clean_audio_path, speaker_timestamps, output_dir
            )
            
            if not initial_speaker_files:
                console.print("❌ No speaker files created from Gemini analysis", style="red")
                return []
            
            # Step 5: Interactive review and refinement
            model_ids = self._interactive_gemini_refinement(initial_speaker_files, speaker_timestamps)
            
            return model_ids
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _interactive_gemini_refinement(self, speaker_files: List[str], 
                                     original_timestamps: Dict[str, List[Tuple[str, str]]]) -> List[str]:
        """Interactive refinement process with Gemini"""
        console.print(f"\n🎭 [bold]Review and refine speakers...[/bold]", style="blue")
        model_ids = []
        
        for speaker_file in speaker_files:
            # Get speaker name from filename
            filename = os.path.basename(speaker_file)
            original_speaker_name = filename.replace("speaker_", "").replace(".wav", "").replace("_", " ")
            
            console.print(f"\n🎤 [bold]Processing: {original_speaker_name}[/bold]", style="cyan")
            
            # Ask user if they want to keep this speaker
            if not Confirm.ask(f"Keep {original_speaker_name}?", default=True):
                console.print("⏭️ Skipping speaker", style="yellow")
                continue
            
            # Let user edit the character name and description
            character_name = Prompt.ask("Character name", default=original_speaker_name.split("(")[0].strip())
            description = Prompt.ask("Character description", default=original_speaker_name)
            
            # Initial refinement with Gemini
            console.print(f"🤖 Asking Gemini to refine timestamps for [bold]{character_name}[/bold]...", style="cyan")
            initial_refined_timestamps = self.gemini_analyzer.refine_speaker(speaker_file, character_name, description)
            
            if not initial_refined_timestamps:
                console.print(f"⚠️ Gemini couldn't find confident segments for {character_name}", style="yellow")
                
                # Ask if user wants to keep the original version
                if Confirm.ask("Keep original version anyway?", default=False):
                    initial_refined_timestamps = original_timestamps.get(original_speaker_name, [])
                else:
                    continue
            
            # Interactive chat refinement
            if Confirm.ask(f"Do you want to further refine {character_name} with interactive chat?", default=True):
                refined_timestamps = self.gemini_analyzer.interactive_refinement_chat(
                    speaker_file, character_name, description, initial_refined_timestamps
                )
            else:
                refined_timestamps = initial_refined_timestamps
            
            # Create final refined audio
            if refined_timestamps:
                console.print(f"🔧 Creating refined audio for [bold]{character_name}[/bold]...", style="cyan")
                final_audio = self.audio_processor.extract_audio_segments_from_timestamps(
                    speaker_file, refined_timestamps, character_name
                )
                
                if len(final_audio) > 5000:  # At least 5 seconds
                    # Save final audio
                    final_audio_path = os.path.join(self.temp_dir, f"final_{create_safe_filename(character_name)}.wav")
                    final_audio.export(final_audio_path, format="wav")
                    
                    # Create Fish Audio model
                    model_id = self.create_fish_model(final_audio_path, character_name, description)
                    if model_id:
                        model_ids.append(model_id)
                        console.print(f"✅ Created model for [bold]{character_name}[/bold]: {model_id}", style="green")
                else:
                    console.print(f"⚠️ Not enough refined audio for {character_name} ({len(final_audio)/1000:.1f}s)", style="yellow")
        
        return model_ids
    
    def _get_target_speakers(self) -> List[str]:
        """Ask user for specific speakers to target"""
        console.print("\n🎯 [bold]Target Specific Speakers (Optional)[/bold]", style="blue")
        console.print("You can specify which characters/speakers you want to extract.")
        console.print("Example: 'Harry Potter', 'Hermione', 'Dumbledore'")
        console.print("Leave empty to detect all speakers automatically.\n")
        
        if not Confirm.ask("Do you want to target specific speakers?", default=False):
            return []
        
        target_speakers = []
        while True:
            speaker = Prompt.ask("Enter speaker/character name (or 'done' to finish)", default="").strip()
            
            if not speaker or speaker.lower() == 'done':
                break
                
            target_speakers.append(speaker)
            console.print(f"✅ Added: {speaker}", style="green")
            
            if not Confirm.ask("Add another speaker?", default=False):
                break
        
        if target_speakers:
            console.print(f"🎯 Will target: {', '.join(target_speakers)}", style="cyan")
        
        return target_speakers