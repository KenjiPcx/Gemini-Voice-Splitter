"""
Audio processing utilities for voice training
"""

import os
import sys
import subprocess
import shutil
import hashlib
from typing import List, Tuple, Optional, Dict
import warnings
from pydub import AudioSegment
import webrtcvad
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.helpers import create_safe_filename

console = Console()


class AudioProcessor:
    """Handles all audio processing operations"""
    
    def __init__(self):
        pass
    
    def convert_to_mp3_for_gemini(self, audio_file: str) -> str:
        """Convert audio file to MP3 format for Gemini API compatibility"""
        try:
            # Check if already MP3
            if audio_file.lower().endswith('.mp3'):
                console.print(f"✅ Audio already in MP3 format: {os.path.basename(audio_file)}", style="green")
                return audio_file
            
            console.print(f"🔄 Converting to MP3 for Gemini compatibility: {os.path.basename(audio_file)}", style="cyan")
            
            # Load audio and convert to MP3
            audio = AudioSegment.from_file(audio_file)
            
            # Create MP3 filename
            base_name = os.path.splitext(audio_file)[0]
            mp3_file = f"{base_name}.mp3"
            
            # Export as MP3 with good quality settings
            audio.export(
                mp3_file,
                format="mp3",
                bitrate="192k",  # Good quality for speech
                parameters=["-ar", "44100"]  # Standard sample rate
            )
            
            file_size_mb = os.path.getsize(mp3_file) / (1024 * 1024)
            console.print(f"✅ Converted to MP3: {os.path.basename(mp3_file)} ({file_size_mb:.1f} MB)", style="green")
            
            return mp3_file
            
        except Exception as e:
            console.print(f"❌ Error converting to MP3: {e}", style="red")
            console.print("⚠️ Using original file format", style="yellow")
            return audio_file
    
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
    
    def download_youtube_audio(self, url: str, video_index: int, downloads_dir: str, temp_dir: str) -> str:
        """Download audio from YouTube URL"""
        # Create hash of URL for consistent filenames
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"video_{video_index}_{url_hash}.wav"
        output_path = os.path.join(downloads_dir, filename)
        
        # Check if already downloaded
        if os.path.exists(output_path):
            console.print(f"✅ Using cached download: {filename}", style="green")
            # Copy to temp dir for processing
            temp_path = os.path.join(temp_dir, f"input_audio_{video_index}.wav")
            shutil.copy2(output_path, temp_path)
            return temp_path
        
        # Download to persistent location
        download_template = os.path.join(downloads_dir, f"video_{video_index}_{url_hash}.%(ext)s")
        
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
                temp_path = os.path.join(temp_dir, f"input_audio_{video_index}.wav")
                shutil.copy2(output_path, temp_path)
                return temp_path
            else:
                raise FileNotFoundError("Downloaded audio file not found")
                
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Error downloading YouTube video: {e.stderr}", style="red")
            sys.exit(1)
    
    def download_all_sources(self, input_sources: List[str], downloads_dir: str, temp_dir: str) -> List[str]:
        """Download all audio sources first"""
        audio_files = []
        
        console.print(f"\n📥 [bold]Downloading {len(input_sources)} audio sources...[/bold]", style="blue")
        
        for i, input_source in enumerate(input_sources):
            console.print(f"\n📹 [bold]Source {i + 1}/{len(input_sources)}[/bold]: [blue]{input_source}[/blue]")
            
            if input_source.startswith(('http://', 'https://')):
                # Download YouTube video
                audio_path = self.download_youtube_audio(input_source, i, downloads_dir, temp_dir)
            else:
                # Copy local file to temp directory
                filename = f"input_audio_{i}.wav"
                audio_path = os.path.join(temp_dir, filename)
                shutil.copy2(input_source, audio_path)
                console.print(f"✅ Copied local file: {os.path.basename(input_source)}", style="green")
            
            audio_files.append(audio_path)
        
        return audio_files
    
    def remove_silence_with_vad(self, audio: AudioSegment) -> AudioSegment:
        """Remove silence using Voice Activity Detection"""
        try:
            # Export audio to raw format for VAD
            raw_audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_data = raw_audio.raw_data
            
            # Initialize VAD
            vad = webrtcvad.Vad(1)  # Aggressiveness level 1 (0-3)
            
            # Process audio in 30ms chunks
            frame_duration = 30  # ms
            frame_size = int(16000 * frame_duration / 1000) * 2  # bytes
            
            # Keep track of voice segments
            voice_segments = []
            current_segment_start = None
            
            for i in range(0, len(audio_data), frame_size):
                frame = audio_data[i:i + frame_size]
                
                if len(frame) < frame_size:
                    break
                
                # Check if frame contains speech
                is_speech = vad.is_speech(frame, 16000)
                
                if is_speech:
                    if current_segment_start is None:
                        current_segment_start = i * 1000 // (16000 * 2)  # Convert to ms
                else:
                    if current_segment_start is not None:
                        segment_end = i * 1000 // (16000 * 2)
                        voice_segments.append((current_segment_start, segment_end))
                        current_segment_start = None
            
            # Handle case where speech continues to the end
            if current_segment_start is not None:
                voice_segments.append((current_segment_start, len(audio)))
            
            if not voice_segments:
                return audio  # Return original if no speech detected
            
            # Combine voice segments with small buffers
            result_audio = AudioSegment.empty()
            buffer_ms = 500  # 0.5 second buffer
            
            for start_ms, end_ms in voice_segments:
                # Add buffer before and after
                buffered_start = max(0, start_ms - buffer_ms)
                buffered_end = min(len(audio), end_ms + buffer_ms)
                
                segment = audio[buffered_start:buffered_end]
                if len(result_audio) > 0:
                    result_audio += AudioSegment.silent(duration=buffer_ms)
                result_audio += segment
            
            return result_audio
            
        except Exception as e:
            console.print(f"⚠️ VAD failed: {e}, keeping original audio", style="yellow")
            return audio
    
    def stitch_all_audio(self, audio_files: List[str], temp_dir: str) -> str:
        """Stitch together multiple audio files with VAD preprocessing"""
        console.print(f"\n🔗 [bold]Stitching {len(audio_files)} audio files with VAD...[/bold]", style="blue")
        
        combined_audio = AudioSegment.empty()
        silence_buffer = AudioSegment.silent(duration=500)  # 0.5 second buffer between files
        
        total_original_duration = 0
        total_processed_duration = 0
        
        for i, audio_file in enumerate(audio_files):
            try:
                console.print(f"📄 Processing file {i+1}: {os.path.basename(audio_file)}")
                
                # Load audio
                audio = AudioSegment.from_file(audio_file)
                total_original_duration += len(audio) / 1000
                
                # Remove silence using VAD
                cleaned_audio = self.remove_silence_with_vad(audio)
                total_processed_duration += len(cleaned_audio) / 1000
                
                # Add to combined audio
                if len(combined_audio) > 0:
                    combined_audio += silence_buffer
                combined_audio += cleaned_audio
                
                console.print(f"  ✅ {len(audio)/1000:.1f}s → {len(cleaned_audio)/1000:.1f}s (VAD cleaned)")
                
                # Add buffer between files (except for the last one)
                if i < len(audio_files) - 1:
                    combined_audio += silence_buffer
                    
            except Exception as e:
                console.print(f"⚠️ Could not load {audio_file}: {e}", style="yellow")
        
        # Save combined audio
        combined_path = os.path.join(temp_dir, "combined_audio.wav")
        combined_audio.export(combined_path, format="wav")
        
        console.print(f"✅ Combined audio: {total_original_duration:.1f}s → {total_processed_duration:.1f}s from {len(audio_files)} sources", style="green")
        return combined_path
    
    def remove_background_music(self, audio_path: str, separator, remove_music: bool, skip_music_separation: bool) -> str:
        """Remove background music and return vocals-only audio"""
        if not remove_music or not separator or skip_music_separation:
            if skip_music_separation:
                console.print("⏭️ Skipping music separation (as requested)", style="yellow")
            return audio_path
            
        console.print("🎵 Removing background music...", style="cyan")
        
        try:
            # Use audio separator to isolate vocals
            output_dir = os.path.dirname(audio_path)
            separator_outputs = separator.separate(audio_path, output_dir)
            
            # Look for vocals file
            vocals_file = None
            for output_file in separator_outputs:
                if 'vocals' in os.path.basename(output_file).lower():
                    vocals_file = output_file
                    break
            
            if vocals_file and os.path.exists(vocals_file):
                console.print("✅ Background music removed", style="green")
                return vocals_file
            else:
                console.print("⚠️ Could not isolate vocals, using original audio", style="yellow")
                return audio_path
                
        except Exception as e:
            console.print(f"❌ Error removing music: {e}", style="red")
            console.print("⚠️ Using original audio", style="yellow")
            return audio_path
    
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
            
            # Stitch segments together with smooth transitions
            console.print(f"🔗 Stitching {len(segments)} segments with smooth transitions...", style="cyan")
            
            if len(segments) == 1:
                # Single segment - just apply fade in/out
                final_audio = segments[0]
                final_audio = final_audio.fade_in(50).fade_out(50)  # 50ms fades
            else:
                # Multiple segments - use crossfading for smooth transitions
                final_audio = segments[0].fade_in(50)  # Fade in first segment
                
                for i, segment in enumerate(segments[1:], 1):
                    # Apply fade out to current audio and fade in to new segment
                    segment = segment.fade_in(50).fade_out(50)
                    
                    # Add intelligent spacing based on segment context
                    # For quality voice training, add minimal natural pause
                    pause_duration = 200  # 0.2 second natural pause
                    natural_pause = AudioSegment.silent(duration=pause_duration)
                    
                    # Smooth concatenation: crossfade the end of final_audio with beginning of segment
                    crossfade_duration = 25  # 25ms crossfade to avoid clicks
                    
                    # Add natural pause and crossfade
                    final_audio = final_audio + natural_pause
                    final_audio = final_audio.append(segment, crossfade=crossfade_duration)
                    
                    console.print(f"  ✅ Smoothly added segment {i+1}/{len(segments)}", style="dim green")
                
                # Final fade out
                final_audio = final_audio.fade_out(50)
            
            console.print(f"✅ [bold]{speaker_name}[/bold]: {total_extracted_duration:.1f}s extracted, {len(final_audio)/1000:.1f}s final (smooth)", style="green")
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
                safe_speaker_name = create_safe_filename(speaker_name)
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
    
    def process_audio_files(self, audio_files: List[str], temp_dir: str, remove_music: bool, 
                           separator, skip_music_separation: bool, pipeline) -> List[str]:
        """Process audio files using pyannote pipeline"""
        # This would contain the main audio processing logic from the original process method
        # For now, return empty list as placeholder
        console.print("🔬 Processing with pyannote...", style="cyan")
        return []