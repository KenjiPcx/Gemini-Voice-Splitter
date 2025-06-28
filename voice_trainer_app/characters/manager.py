"""Character management system for persistent voice training data."""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt

from ..utils.helpers import create_safe_filename

console = Console()

@dataclass
class CharacterMetadata:
    """Metadata for a character's voice training data."""
    name: str
    safe_name: str
    description: str
    created_at: str
    updated_at: str
    clip_count: int
    total_duration: float
    voice_characteristics: List[str]
    source_files: List[str]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CharacterMetadata':
        return cls(**data)
    
    def to_dict(self) -> dict:
        return asdict(self)

class CharacterManager:
    """Manages persistent character folders and metadata."""
    
    def __init__(self, base_dir: str = "character_library"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_dir / "characters.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load character metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.characters = {
                        name: CharacterMetadata.from_dict(meta) 
                        for name, meta in data.items()
                    }
            except Exception as e:
                console.print(f"⚠️ Error loading character metadata: {e}", style="yellow")
                self.characters = {}
        else:
            self.characters = {}
    
    def _save_metadata(self) -> None:
        """Save character metadata to disk."""
        try:
            data = {name: meta.to_dict() for name, meta in self.characters.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"⚠️ Error saving character metadata: {e}", style="yellow")
    
    def get_character_folder(self, character_name: str) -> Path:
        """Get the folder path for a character."""
        safe_name = create_safe_filename(character_name)
        return self.base_dir / safe_name
    
    def character_exists(self, character_name: str) -> bool:
        """Check if a character already exists."""
        return character_name in self.characters
    
    def get_existing_characters(self) -> List[str]:
        """Get list of existing character names."""
        return list(self.characters.keys())
    
    def create_character(self, name: str, description: str = "", voice_characteristics: List[str] = None) -> CharacterMetadata:
        """Create a new character with folder and metadata."""
        if voice_characteristics is None:
            voice_characteristics = []
        
        safe_name = create_safe_filename(name)
        character_folder = self.get_character_folder(name)
        character_folder.mkdir(exist_ok=True)
        
        # Create clips subfolder
        clips_folder = character_folder / "clips"
        clips_folder.mkdir(exist_ok=True)
        
        now = datetime.now().isoformat()
        metadata = CharacterMetadata(
            name=name,
            safe_name=safe_name,
            description=description,
            created_at=now,
            updated_at=now,
            clip_count=0,
            total_duration=0.0,
            voice_characteristics=voice_characteristics,
            source_files=[]
        )
        
        self.characters[name] = metadata
        self._save_metadata()
        
        console.print(f"✅ Created character folder: {character_folder}", style="green")
        return metadata
    
    def add_clip_to_character(self, character_name: str, audio_file_path: str, 
                            source_info: str = "", duration: float = 0.0) -> str:
        """Add an audio clip to a character's collection."""
        if character_name not in self.characters:
            raise ValueError(f"Character '{character_name}' does not exist")
        
        character_folder = self.get_character_folder(character_name)
        clips_folder = character_folder / "clips"
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(audio_file_path).name
        name_part, ext = os.path.splitext(original_filename)
        new_filename = f"{timestamp}_{name_part}{ext}"
        
        destination = clips_folder / new_filename
        
        # Copy the file
        shutil.copy2(audio_file_path, destination)
        
        # Update metadata
        metadata = self.characters[character_name]
        metadata.clip_count += 1
        metadata.total_duration += duration
        metadata.updated_at = datetime.now().isoformat()
        
        if source_info and source_info not in metadata.source_files:
            metadata.source_files.append(source_info)
        
        self._save_metadata()
        
        console.print(f"✅ Added clip to {character_name}: {new_filename}", style="green")
        return str(destination)
    
    def get_character_clips(self, character_name: str) -> List[Path]:
        """Get all audio clips for a character."""
        if character_name not in self.characters:
            return []
        
        clips_folder = self.get_character_folder(character_name) / "clips"
        if not clips_folder.exists():
            return []
        
        # Return all audio files, sorted by creation time
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        clips = [
            f for f in clips_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]
        return sorted(clips, key=lambda x: x.stat().st_mtime)
    
    def prompt_character_selection(self, available_speakers: List[str]) -> Dict[str, str]:
        """Interactive prompt to map detected speakers to characters."""
        console.print("\n🎭 [bold]Character Assignment[/bold]", style="blue")
        console.print("Map detected speakers to character names:")
        
        existing_chars = self.get_existing_characters()
        if existing_chars:
            console.print(f"\n📚 Existing characters: {', '.join(existing_chars)}", style="cyan")
        
        speaker_to_character = {}
        
        for speaker in available_speakers:
            console.print(f"\n🎤 Speaker: [bold]{speaker}[/bold]", style="yellow")
            
            # Ask for character name
            while True:
                character_name = Prompt.ask("Character name (or 'skip' to ignore this speaker)").strip()
                
                if character_name.lower() == 'skip':
                    console.print(f"⏭️ Skipping {speaker}", style="yellow")
                    break
                
                if not character_name:
                    console.print("❌ Character name cannot be empty", style="red")
                    continue
                
                # Check if character exists
                if self.character_exists(character_name):
                    # Show existing character info
                    char_info = self.get_character_info(character_name)
                    console.print(f"📋 Existing character info:", style="cyan")
                    console.print(f"   • Current clips: {char_info.clip_count}")
                    console.print(f"   • Total duration: {char_info.total_duration:.1f}s")
                    console.print(f"   • Description: {char_info.description}")
                    
                    if Confirm.ask(f"Add new clip to existing character '{character_name}'?", default=True):
                        speaker_to_character[speaker] = character_name
                        console.print(f"✅ Will add to existing character: {character_name}", style="green")
                        break
                    else:
                        continue
                else:
                    # New character
                    description = Prompt.ask("Character description (optional)", default="").strip()
                    self.create_character(character_name, description)
                    speaker_to_character[speaker] = character_name
                    console.print(f"✅ Created new character: {character_name}", style="green")
                    break
        
        return speaker_to_character
    
    def show_character_library(self) -> None:
        """Display the current character library."""
        if not self.characters:
            console.print("📚 Character library is empty", style="cyan")
            return
        
        table = Table(title="🎭 Character Library")
        table.add_column("Character", style="cyan", no_wrap=True)
        table.add_column("Clips", justify="right", style="green")
        table.add_column("Duration", justify="right", style="yellow")
        table.add_column("Updated", style="blue")
        table.add_column("Description", style="white")
        
        for name, meta in self.characters.items():
            duration_str = f"{meta.total_duration:.1f}s" if meta.total_duration > 0 else "0s"
            updated_date = datetime.fromisoformat(meta.updated_at).strftime("%Y-%m-%d")
            description = meta.description[:50] + "..." if len(meta.description) > 50 else meta.description
            
            table.add_row(
                name,
                str(meta.clip_count),
                duration_str,
                updated_date,
                description
            )
        
        console.print(table)
    
    def get_character_info(self, character_name: str) -> Optional[CharacterMetadata]:
        """Get metadata for a specific character."""
        return self.characters.get(character_name)
    
    def delete_character(self, character_name: str) -> bool:
        """Delete a character and all their data."""
        if character_name not in self.characters:
            return False
        
        character_folder = self.get_character_folder(character_name)
        if character_folder.exists():
            shutil.rmtree(character_folder)
        
        del self.characters[character_name]
        self._save_metadata()
        
        console.print(f"🗑️ Deleted character: {character_name}", style="red")
        return True
    
    def combine_character_clips(self, character_name: str, output_path: Optional[str] = None) -> Optional[str]:
        """Combine all clips for a character into a single audio file."""
        if character_name not in self.characters:
            console.print(f"❌ Character '{character_name}' not found", style="red")
            return None
        
        clips = self.get_character_clips(character_name)
        if not clips:
            console.print(f"❌ No clips found for '{character_name}'", style="red")
            return None
        
        if len(clips) == 1:
            console.print(f"ℹ️ Only one clip exists for '{character_name}', no combination needed", style="cyan")
            return str(clips[0])
        
        try:
            from pydub import AudioSegment
            
            console.print(f"🔗 Combining {len(clips)} clips for {character_name}...", style="cyan")
            
            # Start with first clip
            combined = AudioSegment.from_file(str(clips[0]))
            
            # Add remaining clips with small gaps
            for clip_path in clips[1:]:
                # Add 0.5 second silence between clips
                combined += AudioSegment.silent(duration=500)
                combined += AudioSegment.from_file(str(clip_path))
            
            # Determine output path
            if output_path is None:
                character_folder = self.get_character_folder(character_name)
                output_path = character_folder / f"combined_{create_safe_filename(character_name)}.wav"
            
            # Export combined audio
            combined.export(output_path, format="wav")
            
            console.print(f"✅ Combined audio saved: {output_path}", style="green")
            console.print(f"📊 Total duration: {len(combined)/1000:.1f}s", style="cyan")
            
            return str(output_path)
            
        except Exception as e:
            console.print(f"❌ Error combining clips: {e}", style="red")
            return None