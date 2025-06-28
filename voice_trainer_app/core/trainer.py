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
from ..characters.manager import CharacterManager
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
        self.character_manager = CharacterManager()
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
        """Create a Fish Audio model from a single audio file"""
        url = "https://api.fish.audio/model"
        
        headers = {
            "Authorization": f"Bearer {self.fish_api_key}"
        }
        
        console.print(f"🐟 Creating Fish Audio model for: [bold]{character_name}[/bold]", style="cyan")
        
        # Prepare the form data
        with open(audio_file, 'rb') as f:
            files = {
                'voices': (f"{character_name}.wav", f, 'audio/wav')
            }
            data = {
                'title': character_name,
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
                    console.print(f"✅ Created model '[bold]{character_name}[/bold]' with ID: [green]{model_id}[/green]")
                    return model_id
                else:
                    console.print(f"❌ Failed to get model ID for '[bold]{character_name}[/bold]'", style="red")
                    return None
                    
            except requests.exceptions.RequestException as e:
                console.print(f"❌ Error creating model '[bold]{character_name}[/bold]': {e}", style="red")
                return None
    
    def create_fish_model_from_multiple_files(self, audio_files: List[str], character_name: str, description: str) -> Optional[str]:
        """Create a Fish Audio model from multiple audio files"""
        url = "https://api.fish.audio/model"
        
        headers = {
            "Authorization": f"Bearer {self.fish_api_key}"
        }
        
        console.print(f"🐟 Creating Fish Audio model for: [bold]{character_name}[/bold] from {len(audio_files)} clips", style="cyan")
        
        # Prepare the form data with multiple files
        files = []
        data = {
            'title': character_name,
            'description': description,
            'train_mode': 'fast',
            'enhance_audio_quality': 'true'
        }
        
        try:
            # Open all audio files
            file_handles = []
            for i, audio_file in enumerate(audio_files):
                f = open(audio_file, 'rb')
                file_handles.append(f)
                files.append(('voices', (f"{character_name}_{i}.wav", f, 'audio/wav')))
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                model_id = result.get('id')
                
                if model_id:
                    console.print(f"✅ Created model '[bold]{character_name}[/bold]' with ID: [green]{model_id}[/green]")
                    return model_id
                else:
                    console.print(f"❌ Failed to get model ID for '[bold]{character_name}[/bold]'", style="red")
                    return None
                    
            finally:
                # Close all file handles
                for f in file_handles:
                    f.close()
                    
        except requests.exceptions.RequestException as e:
            console.print(f"❌ Error creating model '[bold]{character_name}[/bold]': {e}", style="red")
            return None
        except Exception as e:
            console.print(f"❌ Unexpected error creating model '[bold]{character_name}[/bold]': {e}", style="red")
            return None
    
    def process(self, input_sources: List[str]) -> List[str]:
        """Main processing function - processes each source individually using pyannote"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup pipelines
            self.setup_pipeline()
            self.setup_music_separator()
            
            all_model_ids = []
            
            # Process each source individually
            for i, input_source in enumerate(input_sources, 1):
                console.print(f"\n📹 [bold]Processing source {i}/{len(input_sources)}: {input_source}[/bold]", style="blue")
                
                # Step 1: Download this audio source
                audio_files = self.audio_processor.download_all_sources(
                    [input_source], self.downloads_dir, self.temp_dir
                )
                
                if not audio_files:
                    console.print(f"⚠️ Failed to download source {i}, skipping...", style="yellow")
                    continue
                
                # Step 2: Process this audio file (clean, separate voices)
                audio_file = audio_files[0]  # Single file
                
                # Clean audio (remove music if requested)
                clean_audio_path = self.audio_processor.remove_background_music(
                    audio_file, self.separator, self.remove_music, self.skip_music_separation
                )
                
                # Apply voice activity detection
                console.print("🔇 Applying voice activity detection...", style="cyan")
                vad_audio_path = self.audio_processor.apply_vad(clean_audio_path, self.temp_dir)
                
                if not vad_audio_path:
                    console.print(f"⚠️ VAD processing failed for source {i}, skipping...", style="yellow")
                    continue
                
                # Run pyannote speaker separation
                console.print("🔬 Running pyannote speaker separation...", style="cyan")
                speaker_audio_files = self.audio_processor.separate_speakers(
                    vad_audio_path, self.temp_dir, self.pipeline
                )
                
                if not speaker_audio_files:
                    console.print(f"⚠️ No speakers detected in source {i}, skipping...", style="yellow")
                    continue
                
                console.print(f"✅ pyannote found {len(speaker_audio_files)} speakers", style="green")
                
                # Get speaker names from filenames for character assignment
                speaker_names = []
                speaker_file_map = {}
                
                for speaker_file in speaker_audio_files:
                    filename = os.path.basename(speaker_file)
                    speaker_name = filename.replace("speaker_", "").replace(".wav", "").replace("_", " ")
                    speaker_names.append(speaker_name)
                    speaker_file_map[speaker_name] = speaker_file
                
                # Show current character library first
                self.character_manager.show_character_library()
                
                # Use character manager for batch assignment
                speaker_to_character = self.character_manager.prompt_character_selection(speaker_names)
                
                # Process each assigned speaker
                source_model_ids = []
                for speaker_name, character_name in speaker_to_character.items():
                    speaker_file = speaker_file_map[speaker_name]
                    
                    console.print(f"\n🎤 [bold]Processing: {speaker_name} → {character_name}[/bold]", style="cyan")
                    
                    # Get character info for description
                    character_info = self.character_manager.get_character_info(character_name)
                    description = character_info.description if character_info else character_name
                    
                    # Add clip to character folder
                    try:
                        # Get audio duration
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_wav(speaker_file)
                        duration_seconds = len(audio_segment) / 1000
                        
                        source_info = f"pyannote separation - source {i}"
                        saved_clip_path = self.character_manager.add_clip_to_character(
                            character_name, speaker_file, source_info, duration_seconds
                        )
                        console.print(f"💾 Saved to character library: {saved_clip_path}", style="green")
                        
                        # Create Fish Audio model using all clips for this character
                        console.print(f"🐟 Creating Fish model for [bold]{character_name}[/bold]...", style="blue")
                        
                        # Get all clips for this character
                        character_clips = self.character_manager.get_character_clips(character_name)
                        
                        if character_clips:
                            # Convert Path objects to strings
                            clip_paths = [str(clip) for clip in character_clips]
                            
                            # Create model with multiple files
                            model_id = self.create_fish_model_from_multiple_files(clip_paths, character_name, description)
                            
                            if model_id:
                                source_model_ids.append(model_id)
                                console.print(f"✅ Created model for [bold]{character_name}[/bold]: {model_id}", style="green")
                                console.print(f"🎤 Used {len(clip_paths)} audio clips", style="cyan")
                            else:
                                console.print(f"❌ Failed to create model for {character_name}", style="red")
                        else:
                            console.print(f"⚠️ No clips found for {character_name}", style="yellow")
                            
                    except Exception as e:
                        console.print(f"⚠️ Error processing {character_name}: {e}", style="yellow")
                
                all_model_ids.extend(source_model_ids)
                console.print(f"📊 Completed source {i}: {len(source_model_ids)} models created", style="green")
            
            # Summary
            if all_model_ids:
                console.print(f"\n🎉 [bold]Processing complete! Created {len(all_model_ids)} models total[/bold]", style="green")
            else:
                console.print(f"\n⚠️ [bold]No models were created from any sources[/bold]", style="yellow")
            
            return all_model_ids
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def process_with_gemini(self, input_sources: List[str]) -> List[str]:
        """Main processing function - processes each source individually using Gemini AI"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup music separator only
            self.setup_music_separator()
            
            if not self.gemini_analyzer:
                console.print("❌ Gemini is not enabled. Use --gemini-api-key to enable.", style="red")
                return []
            
            # Ask for analysis instructions (optional)
            target_speakers, instructions = self._get_analysis_instructions(input_sources)
            
            all_model_ids = []
            
            # Process each source individually
            for i, input_source in enumerate(input_sources, 1):
                console.print(f"\n📹 [bold]Processing source {i}/{len(input_sources)}: {input_source}[/bold]", style="blue")
                
                # Step 1: Download this audio source
                audio_files = self.audio_processor.download_all_sources(
                    [input_source], self.downloads_dir, self.temp_dir
                )
                
                if not audio_files:
                    console.print(f"⚠️ Failed to download source {i}, skipping...", style="yellow")
                    continue
                
                audio_file = audio_files[0]  # Single file
                
                # Step 2: Clean audio (remove music if requested)
                clean_audio_path = self.audio_processor.remove_background_music(
                    audio_file, self.separator, self.remove_music, self.skip_music_separation
                )
                
                # Step 3: Analyze with Gemini
                console.print(f"🤖 [bold]Analyzing audio with Gemini...[/bold]", style="cyan")
                if target_speakers:
                    console.print(f"🎯 Targeting specific speakers: {', '.join(target_speakers)}", style="cyan")
                speaker_timestamps = self.gemini_analyzer.analyze_audio(clean_audio_path, target_speakers, instructions)
                
                if not speaker_timestamps:
                    console.print(f"⚠️ Gemini could not identify speakers in source {i}, skipping...", style="yellow")
                    continue
                
                console.print(f"✅ Gemini found {len(speaker_timestamps)} speakers", style="green")
                
                # Step 4: Create initial speaker files
                output_dir = os.path.join(self.temp_dir, f"gemini_speakers_source_{i}")
                initial_speaker_files = self.audio_processor.create_speaker_files_from_timestamps(
                    clean_audio_path, speaker_timestamps, output_dir
                )
                
                if not initial_speaker_files:
                    console.print(f"⚠️ No speaker files created from source {i}, skipping...", style="yellow")
                    continue
                
                # Step 5: Interactive review and refinement for this source
                source_model_ids = self._interactive_gemini_refinement(initial_speaker_files, speaker_timestamps)
                all_model_ids.extend(source_model_ids)
                
                console.print(f"📊 Completed source {i}: {len(source_model_ids)} models created", style="green")
            
            # Summary
            if all_model_ids:
                console.print(f"\n🎉 [bold]Processing complete! Created {len(all_model_ids)} models total[/bold]", style="green")
            else:
                console.print(f"\n⚠️ [bold]No models were created from any sources[/bold]", style="yellow")
            
            return all_model_ids
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _get_analysis_instructions(self, input_sources: List[str]) -> tuple[List[str], str]:
        """Ask user for analysis instructions including target speakers and context"""
        console.print("\n🎯 [bold]Analysis Instructions (Optional)[/bold]", style="blue")
        console.print("Provide instructions to help Gemini analyze the audio.")
        
        # Show source list
        if len(input_sources) == 1:
            console.print(f"Source: {input_sources[0]}", style="dim")
        else:
            console.print(f"Sources: {len(input_sources)} files/URLs", style="dim")
        
        console.print("\nExamples:")
        console.print("  📺 Context: 'Harry Potter movie scene'")
        console.print("  🎯 Target speakers: 'Harry Potter, Hermione'")
        console.print("  📝 Combined: 'Harry Potter movie, extract Harry and Hermione voices'")
        console.print("  📋 Other: 'Focus on clear dialogue, ignore background voices'")
        
        if not Confirm.ask("\nAdd instructions to help Gemini?", default=True):
            return [], ""
        
        instructions = Prompt.ask("Instructions (context, target speakers, etc)", default="").strip()
        
        # Try to extract target speakers from instructions
        target_speakers = []
        if instructions:
            # Look for speaker names in the instructions
            # Simple heuristic: look for comma-separated names after keywords
            lower_instructions = instructions.lower()
            if any(keyword in lower_instructions for keyword in ['target', 'extract', 'focus on', 'get']):
                # Try to extract names (this is a simple heuristic)
                words = instructions.split()
                potential_speakers = []
                capture_next = False
                
                for i, word in enumerate(words):
                    clean_word = word.strip(',.:;"\'').strip()
                    if capture_next and clean_word and clean_word[0].isupper():
                        # Check if it might be a name (starts with capital)
                        potential_speakers.append(clean_word)
                    
                    if word.lower() in ['target', 'extract', 'focus', 'get']:
                        capture_next = True
                    elif word in [',', 'and', '&']:
                        capture_next = True
                    else:
                        capture_next = False
                
                if potential_speakers:
                    target_speakers = potential_speakers[:5]  # Limit to 5 names
        
        if instructions:
            console.print(f"📝 Instructions: {instructions}", style="cyan")
            if target_speakers:
                console.print(f"🎯 Detected targets: {', '.join(target_speakers)}", style="dim cyan")
        
        return target_speakers, instructions
    
    def _interactive_gemini_refinement(self, speaker_files: List[str], 
                                     original_timestamps: Dict[str, List[Tuple[str, str]]]) -> List[str]:
        """Interactive refinement process with Gemini"""
        console.print(f"\n🎭 [bold]Review and refine speakers...[/bold]", style="blue")
        model_ids = []
        
        # First, get all speaker names for character assignment
        all_speakers = []
        speaker_file_map = {}
        
        for speaker_file in speaker_files:
            # Get speaker name from filename
            filename = os.path.basename(speaker_file)
            original_speaker_name = filename.replace("speaker_", "").replace(".wav", "").replace("_", " ")
            all_speakers.append(original_speaker_name)
            speaker_file_map[original_speaker_name] = speaker_file
        
        # Show current character library first
        self.character_manager.show_character_library()
        
        # Use character manager for batch assignment
        speaker_to_character = self.character_manager.prompt_character_selection(all_speakers)
        
        # Process each assigned speaker
        for original_speaker_name, character_name in speaker_to_character.items():
            speaker_file = speaker_file_map[original_speaker_name]
            
            console.print(f"\n🎤 [bold]Processing: {original_speaker_name} → {character_name}[/bold]", style="cyan")
            
            # Get character info for description
            character_info = self.character_manager.get_character_info(character_name)
            description = character_info.description if character_info else character_name
            
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
                    # Save final audio to temp first
                    final_audio_path = os.path.join(self.temp_dir, f"final_{create_safe_filename(character_name)}.wav")
                    final_audio.export(final_audio_path, format="wav")
                    
                    # Add clip to character folder
                    try:
                        duration_seconds = len(final_audio) / 1000
                        source_info = f"Gemini analysis - {len(refined_timestamps)} segments"
                        saved_clip_path = self.character_manager.add_clip_to_character(
                            character_name, final_audio_path, source_info, duration_seconds
                        )
                        console.print(f"💾 Saved to character library: {saved_clip_path}", style="green")
                    except Exception as e:
                        console.print(f"⚠️ Could not save to character library: {e}", style="yellow")
                    
                    # Create Fish Audio model using all clips for this character
                    console.print(f"🐟 Creating Fish model for [bold]{character_name}[/bold]...", style="blue")
                    
                    # Get all clips for this character
                    character_clips = self.character_manager.get_character_clips(character_name)
                    
                    if character_clips:
                        # Convert Path objects to strings
                        clip_paths = [str(clip) for clip in character_clips]
                        
                        # Check if we have create_model_from_multiple_files method
                        try:
                            model_id = self.create_fish_model_from_multiple_files(clip_paths, character_name, description)
                        except AttributeError:
                            # Fallback to single file if multiple file method doesn't exist
                            model_id = self.create_fish_model(final_audio_path, character_name, description)
                        
                        if model_id:
                            model_ids.append(model_id)
                            console.print(f"✅ Created model for [bold]{character_name}[/bold]: {model_id}", style="green")
                            console.print(f"🎤 Used {len(clip_paths)} audio clips", style="cyan")
                        else:
                            console.print(f"❌ Failed to create model for {character_name}", style="red")
                    else:
                        console.print(f"⚠️ No clips found for {character_name}", style="yellow")
                else:
                    console.print(f"⚠️ Not enough refined audio for {character_name} ({len(final_audio)/1000:.1f}s)", style="yellow")
        
        return model_ids
    
