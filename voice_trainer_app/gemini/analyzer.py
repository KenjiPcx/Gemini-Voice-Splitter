"""
Gemini AI integration for audio analysis and speaker separation
"""

import json
import os
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
import google.generativeai as genai
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()


class TimestampPair(BaseModel):
    """Represents a start and end timestamp"""
    start: str = Field(description="Start timestamp in MM:SS format")
    end: str = Field(description="End timestamp in MM:SS format")


class Speaker(BaseModel):
    """Represents a speaker with their voice characteristics and timestamps"""
    name: str = Field(description="Speaker identifier with voice characteristics (e.g., 'Speaker 1 (Deep male voice)')")
    description: str = Field(description="Detailed voice characteristics")
    timestamps: List[TimestampPair] = Field(description="List of timestamps where this speaker is talking")


class SpeakerAnalysis(BaseModel):
    """Complete speaker analysis result"""
    speakers: List[Speaker] = Field(description="List of identified speakers")
    confidence: str = Field(description="Overall confidence level of the analysis")


class RefinementResult(BaseModel):
    """Result of speaker refinement with high-confidence timestamps"""
    timestamps: List[TimestampPair] = Field(description="High-confidence timestamp segments")
    confidence_level: str = Field(description="Confidence level for the refined segments")
    explanation: str = Field(description="Explanation of what was changed or why these segments were selected")


class GeminiAnalyzer:
    """Handles Gemini AI integration for speaker analysis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self._setup_gemini()
        # Import audio processor for MP3 conversion
        from ..audio.processing import AudioProcessor
        self.audio_processor = AudioProcessor()
    
    def _setup_gemini(self):
        """Initialize Gemini API"""
        try:
            console.print("🤖 Setting up Gemini API...", style="cyan")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            console.print("✅ Gemini API ready", style="green")
        except Exception as e:
            console.print(f"❌ Error setting up Gemini: {e}", style="red")
            self.model = None
    
    def analyze_audio(self, audio_file: str, target_speakers: List[str] = None) -> Dict[str, List[Tuple[str, str]]]:
        """Use Gemini to analyze audio and extract speaker timestamps"""
        if not self.model:
            console.print("❌ Gemini model not available", style="red")
            return {}
        
        try:
            console.print("🤖 Analyzing audio with Gemini...", style="cyan")
            
            # Convert to MP3 for better Gemini compatibility
            mp3_file = self.audio_processor.convert_to_mp3_for_gemini(audio_file)
            
            # Upload audio file to Gemini using Files API
            console.print("📤 Uploading audio to Gemini...", style="cyan")
            file_size = os.path.getsize(mp3_file)
            console.print(f"📊 Audio file size: {file_size / (1024*1024):.1f} MB", style="dim")
            
            audio_file_obj = genai.upload_file(mp3_file)
            console.print(f"✅ Audio uploaded successfully", style="green")
            
            # Show streaming preview first for faster feedback (non-structured)
            if file_size < 50 * 1024 * 1024:  # Only for files under 50MB
                console.print("🔄 Getting quick streaming preview first...", style="yellow")
                self.stream_analysis_preview(audio_file_obj)
                console.print("\n✨ Now performing detailed structured analysis (this will take longer)...", style="cyan")
            
            prompt = self._get_analysis_prompt(target_speakers)
            
            console.print("🧠 Gemini is analyzing the audio...", style="cyan")
            
            # Use status indicator for better UX
            with console.status("[cyan]Gemini processing audio...", spinner="dots"):
                response = self.model.generate_content(
                    [prompt, audio_file_obj],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=SpeakerAnalysis
                    )
                )
            
            # Parse structured response
            try:
                analysis = SpeakerAnalysis.model_validate_json(response.text)
                
                # Convert to the format expected by the rest of the system
                speaker_data = {}
                for speaker in analysis.speakers:
                    timestamps = [(ts.start, ts.end) for ts in speaker.timestamps]
                    speaker_data[speaker.name] = timestamps
                
                console.print(f"✅ Gemini found {len(speaker_data)} speakers with {analysis.confidence} confidence", style="green")
                return speaker_data
                
            except Exception as e:
                console.print(f"❌ Error parsing structured response: {e}", style="red")
                console.print(f"Response was: {response.text}...", style="yellow")
                return {}
                
        except Exception as e:
            console.print(f"❌ Error analyzing audio with Gemini: {e}", style="red")
            return {}
    
    def refine_speaker(self, audio_file: str, speaker_name: str, description: str) -> List[Tuple[str, str]]:
        """Use Gemini to refine timestamps for a specific speaker"""
        if not self.model:
            console.print("❌ Gemini model not available", style="red")
            return []
        
        try:
            console.print(f"🤖 Refining timestamps for [bold]{speaker_name}[/bold] with Gemini...", style="cyan")
            
            # Convert to MP3 and upload audio file to Gemini
            mp3_file = self.audio_processor.convert_to_mp3_for_gemini(audio_file)
            console.print("📤 Uploading speaker audio for refinement...", style="cyan")
            audio_file_obj = genai.upload_file(mp3_file)
            
            prompt = self._get_refinement_prompt(speaker_name, description)
            
            console.print("🧠 Gemini is refining the voice segments...", style="cyan")
            
            # Use status indicator for better UX
            with console.status("[cyan]Gemini refining segments...", spinner="dots"):
                response = self.model.generate_content(
                    [prompt, audio_file_obj],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=RefinementResult
                    )
                )
            
            # Parse structured response
            try:
                refinement = RefinementResult.model_validate_json(response.text)
                timestamps = [(ts.start, ts.end) for ts in refinement.timestamps]
                
                console.print(f"✅ Gemini refined to {len(timestamps)} high-confidence segments ({refinement.confidence_level})", style="green")
                return timestamps
                
            except Exception as e:
                console.print(f"❌ Error parsing structured refinement response: {e}", style="red")
                console.print(f"Response was: {response.text}...", style="yellow")
                return []
                
        except Exception as e:
            console.print(f"❌ Error refining speaker with Gemini: {e}", style="red")
            return []
    
    def _get_analysis_prompt(self, target_speakers: List[str] = None) -> str:
        """Get the prompt for initial audio analysis"""
        base_prompt = """
        Please analyze this audio file and identify different speakers. For each distinct speaker you detect:

        1. Create a clear speaker identifier with voice characteristics (e.g., "Speaker 1 (Deep male voice)", "Speaker 2 (Higher-pitched female voice)")
        2. Provide a detailed description of their voice characteristics
        3. Extract ONLY the clearest, longest segments where that speaker is talking

        CRITICAL REQUIREMENTS FOR HIGH-QUALITY VOICE TRAINING:
        - Extract 30-60 seconds total per speaker (aim for 3-8 segments of 5-15 seconds each)
        - ONLY include segments that are at least 4 seconds long
        - Prioritize clear, uninterrupted speech with natural sentence boundaries
        - Include segments that have natural pauses (0.5+ seconds) between sentences
        - Skip short interjections, "ums", incomplete words, or overlapping speech
        - Focus on segments where the speaker is clearly audible without background noise
        - Quality over quantity - better to have fewer perfect segments than many choppy ones
        - Only include speakers with enough clear speech for quality training
        - Provide an overall confidence level for your analysis (high/medium/low)
        
        Think of this as extracting the BEST examples of each voice, not every word they say.
        """
        
        if target_speakers:
            target_prompt = f"""
        
        IMPORTANT: The user is specifically looking for these characters/speakers:
        {', '.join(target_speakers)}
        
        Please focus your analysis on finding and extracting audio segments for these specific speakers. 
        If you can identify any of these speakers in the audio, prioritize their segments and provide detailed timestamps.
        You may also identify other speakers, but give priority to the requested ones.
        """
            base_prompt += target_prompt
        
        base_prompt += "\n\nThe response will be structured as JSON with speakers, descriptions, timestamps, and confidence level."
        return base_prompt
    
    def _get_refinement_prompt(self, speaker_name: str, description: str) -> str:
        """Get the prompt for speaker refinement"""
        return f"""
        This audio clip should contain the voice of {speaker_name} ({description}). 

        Please extract ONLY the highest quality segments for AI voice training. Focus on:

        QUALITY CRITERIA (be extremely selective):
        1. 100% confident it's {speaker_name} speaking clearly
        2. Segments are at least 4-6 seconds long (complete sentences/phrases)
        3. Clear audio with minimal background noise
        4. Natural speech patterns with proper sentence boundaries
        5. Include natural pauses between sentences (0.5+ seconds)
        6. NO overlapping speech, interruptions, or incomplete words
        7. NO "ums", "ahs", or filler sounds
        8. Speaker sounds natural and conversational (not rushed or distorted)

        TARGET: Extract 30-45 seconds total of the BEST quality segments.
        Better to return fewer perfect segments than many mediocre ones.

        Think of this as creating a voice sample reel - only the clearest, most representative examples.

        Confidence level for your refinement analysis (very_high/high/medium/low).
        """
    
    def interactive_refinement_chat(self, audio_file: str, character_name: str, description: str, 
                                   initial_timestamps: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Interactive chat with Gemini to refine audio segments based on user feedback"""
        if not self.model:
            console.print("❌ Gemini model not available", style="red")
            return initial_timestamps
        
        console.print(f"\n💬 [bold]Interactive Refinement Chat for {character_name}[/bold]", style="blue")
        console.print("You can now chat with Gemini to refine the audio segments!")
        console.print("Examples: 'Focus on the emotional parts', 'Remove segments with background noise', 'Keep only clear dialogue'")
        console.print("Type 'done' when satisfied, or 'restart' to go back to original timestamps.\n")
        
        current_timestamps = initial_timestamps.copy()
        
        try:
            # Convert to MP3 and upload audio file once
            mp3_file = self.audio_processor.convert_to_mp3_for_gemini(audio_file)
            audio_file_obj = genai.upload_file(mp3_file)
            
            while True:
                # Show current segments
                console.print(f"\n📋 [bold]Current segments for {character_name}[/bold] ({len(current_timestamps)} segments):")
                for i, (start, end) in enumerate(current_timestamps, 1):
                    console.print(f"  {i}. {start} - {end}")
                
                total_duration = sum(
                    self._timestamp_to_seconds(end) - self._timestamp_to_seconds(start) 
                    for start, end in current_timestamps
                )
                console.print(f"📊 Total duration: {total_duration:.1f}s\n")
                
                # Get user input
                user_input = Prompt.ask("💬 Chat with Gemini (or 'done'/'restart')")
                
                if user_input.lower() == 'done':
                    if Confirm.ask("Are you satisfied with these segments?", default=True):
                        break
                    else:
                        continue
                
                if user_input.lower() == 'restart':
                    current_timestamps = initial_timestamps.copy()
                    console.print("🔄 Reset to original timestamps", style="yellow")
                    continue
                
                # Chat with Gemini for refinement
                console.print("🤖 Gemini is processing your request...", style="cyan")
                
                chat_prompt = f"""
                You are helping refine audio segments for the character "{character_name}" ({description}).
                
                Current timestamp segments:
                {[f"{start} - {end}" for start, end in current_timestamps]}
                
                User request: "{user_input}"
                
                Based on the user's feedback and the audio content, please provide refined timestamp segments.
                Consider the user's request and adjust the segments accordingly.
                
                Provide your response as JSON with the refined timestamp pairs and an explanation of what you changed.
                """
                
                try:
                    with console.status("[cyan]Gemini thinking...", spinner="dots"):
                        response = self.model.generate_content(
                            [chat_prompt, audio_file_obj],
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json",
                                response_schema=RefinementResult
                            )
                        )
                    
                    refinement = RefinementResult.model_validate_json(response.text)
                    new_timestamps = [(ts.start, ts.end) for ts in refinement.timestamps]
                    
                    if new_timestamps:
                        current_timestamps = new_timestamps
                        console.print(f"✅ Gemini refined segments ({refinement.confidence_level})", style="green")
                        console.print(f"💡 Explanation: {refinement.explanation}", style="dim")
                    else:
                        console.print("⚠️ No changes made to segments", style="yellow")
                
                except Exception as e:
                    console.print(f"❌ Error processing chat: {e}", style="red")
                    console.print("Please try rephrasing your request.", style="yellow")
        
        except Exception as e:
            console.print(f"❌ Error in interactive chat: {e}", style="red")
            return initial_timestamps
        
        return current_timestamps
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert MM:SS timestamp to seconds"""
        try:
            parts = timestamp.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            return 0
        except:
            return 0
    
    def stream_analysis_preview(self, audio_file_obj) -> str:
        """Stream a quick preview analysis of the audio (non-structured)"""
        if not self.model:
            return ""
        
        try:
            prompt = """
            Please provide a quick preview analysis of this audio file.
            Listen to the audio and tell me:
            1. How many distinct speakers you can identify
            2. General characteristics of their voices
            3. Overall audio quality
            
            Keep it brief and conversational.
            """
            
            console.print("🌊 [bold]Gemini Streaming Preview:[/bold]", style="blue")
            
            # Use streaming for real-time response
            response_stream = self.model.generate_content(
                [prompt, audio_file_obj],
                stream=True
            )
            
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    # Print chunk immediately for streaming effect
                    console.print(chunk.text, end="", style="cyan")
                    full_response += chunk.text
            
            console.print()  # New line after streaming
            return full_response
            
        except Exception as e:
            console.print(f"❌ Error in streaming preview: {e}", style="red")
            return ""
    
