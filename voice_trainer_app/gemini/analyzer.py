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
from google import genai
from google.genai import types

console = Console()


class TimestampPair(BaseModel):
    """Represents a start and end timestamp"""
    start: str = Field(description="Start timestamp in MM:SS format, don't do microseconds")
    end: str = Field(description="End timestamp in MM:SS format, don't do microseconds")


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
        self.gemini_client = None
        self._setup_gemini()
        # Import audio processor for MP3 conversion
        from ..audio.processing import AudioProcessor
        self.audio_processor = AudioProcessor()
    
    def _setup_gemini(self):
        """Initialize Gemini API"""
        try:
            console.print("🤖 Setting up Gemini API...", style="cyan")
            # Use a model that supports structured output properly
            self.gemini_client = genai.Client(api_key=self.api_key)
            console.print("✅ Gemini API ready", style="green")
        except Exception as e:
            console.print(f"❌ Error setting up Gemini: {e}", style="red")
            self.gemini_client = None
    
    def analyze_audio(self, audio_file: str, target_speakers: List[str] = None, instructions: str = "") -> Dict[str, List[Tuple[str, str]]]:
        """Use Gemini to analyze audio and extract speaker timestamps"""
        if not self.gemini_client:
            console.print("❌ Gemini client not available", style="red")
            return {}
        
        try:
            console.print("🤖 Analyzing audio with Gemini...", style="cyan")
            
            # Convert to MP3 for better Gemini compatibility
            mp3_file = self.audio_processor.convert_to_mp3_for_gemini(audio_file)
            
            # Upload audio file to Gemini using Files API
            console.print("📤 Uploading audio to Gemini...", style="cyan")
            file_size = os.path.getsize(mp3_file)
            console.print(f"📊 Audio file size: {file_size / (1024*1024):.1f} MB", style="dim")
            
            audio_file_obj = self.gemini_client.files.upload(file=mp3_file)
            console.print(f"✅ Audio uploaded successfully", style="green")
            
            prompt = self._get_analysis_prompt(target_speakers, instructions)
            
            console.print("🧠 Gemini is analyzing the audio...", style="cyan")
            
            # Try structured output first, with fallback to manual JSON
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[prompt, audio_file_obj],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SpeakerAnalysis,
                        thinking_config=types.ThinkingConfig(
                            include_thoughts=True
                        )
                    )
                )
            except Exception as e:
                console.print(f"⚠️ Structured output failed, using manual JSON: {e}", style="yellow")
                # Fallback: manual JSON prompting
                json_prompt = prompt + "\n\nIMPORTANT: Respond with valid JSON only, following this exact schema:\n" + str(SpeakerAnalysis.model_json_schema())
                response = self.gemini_client.models.generate_content(model="gemini-2.5-pro", contents=[json_prompt, audio_file_obj])
            
            # Extract and show thinking summary if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            # Show condensed thinking summary
                            summary = part.text
                            if summary:
                                console.print(f"🧐 [italic]{summary}[/italic]", style="dim yellow")
            
            console.print("✅ Analysis complete", style="green")
            
            # Parse structured response
            try:
                if not response.parsed:
                    console.print("❌ Empty response from Gemini", style="red")
                    return {}
                
                # Debug: show response snippet
                response_preview = str(response.parsed)[:200] + "..." if len(str(response.parsed)) > 200 else str(response.parsed)
                console.print(f"🐛 [dim]Response preview: {response_preview}[/dim]", style="dim yellow")
                
                analysis = SpeakerAnalysis.model_validate(response.parsed.model_dump()) # reparse the response to get pydantic types in code
                
                # Convert to the format expected by the rest of the system
                speaker_data = {}
                for speaker in analysis.speakers:
                    timestamps = [(ts.start, ts.end) for ts in speaker.timestamps]
                    speaker_data[speaker.name] = timestamps
                
                console.print(f"✅ Gemini found {len(speaker_data)} speakers with {analysis.confidence} confidence", style="green")
                return speaker_data
                
            except Exception as e:
                console.print(f"❌ Error parsing JSON response: {e}", style="red")
                console.print(f"Raw response: {response.text[:500]}...", style="dim yellow")
                
                # Try to extract JSON from response manually
                try:
                    import json
                    import re
                    
                    # Look for JSON content
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        analysis = SpeakerAnalysis.model_validate_json(json_str)
                        
                        speaker_data = {}
                        for speaker in analysis.speakers:
                            timestamps = [(ts.start, ts.end) for ts in speaker.timestamps]
                            speaker_data[speaker.name] = timestamps
                        
                        console.print(f"✅ Recovered {len(speaker_data)} speakers from response", style="green")
                        return speaker_data
                        
                except Exception as recovery_error:
                    console.print(f"❌ Recovery failed: {recovery_error}", style="red")
                
                return {}
                
        except Exception as e:
            console.print(f"❌ Error analyzing audio with Gemini: {e}", style="red")
            return {}
    
    def refine_speaker(self, audio_file: str, speaker_name: str, description: str) -> List[Tuple[str, str]]:
        """Use Gemini to refine timestamps for a specific speaker"""
        if not self.gemini_client:
            console.print("❌ Gemini client not available", style="red")
            return []
        
        try:
            console.print(f"🤖 Refining timestamps for [bold]{speaker_name}[/bold] with Gemini...", style="cyan")
            
            # Convert to MP3 and upload audio file to Gemini
            mp3_file = self.audio_processor.convert_to_mp3_for_gemini(audio_file)
            console.print("📤 Uploading speaker audio for refinement...", style="cyan")
            audio_file_obj = self.gemini_client.files.upload(file=mp3_file)
            
            prompt = self._get_refinement_prompt(speaker_name, description)
            
            console.print("🧠 Gemini is refining segments...", style="cyan")
            
            # Try structured output with fallback
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[prompt, audio_file_obj],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RefinementResult
                    )
                )
            except Exception as e:
                console.print(f"⚠️ Structured refinement failed, using manual JSON: {e}", style="yellow")
                json_prompt = prompt + "\n\nIMPORTANT: Respond with valid JSON only, following this exact schema:\n" + str(RefinementResult.model_json_schema())
                response = self.gemini_client.models.generate_content(model="gemini-2.5-pro", contents=[json_prompt, audio_file_obj])
            
            # Extract thinking summary for refinement
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            summary = part.text
                            if summary:
                                console.print(f"🧐 [italic]{summary}[/italic]", style="dim yellow")
            
            console.print("✅ Refinement complete", style="green")
            
            # Parse structured response
            try:
                if not response.parsed:
                    console.print("❌ Empty refinement response from Gemini", style="red")
                    return []
                
                refinement = RefinementResult.model_validate(response.parsed.model_dump())
                timestamps = [(ts.start, ts.end) for ts in refinement.timestamps]
                
                console.print(f"✅ Gemini refined to {len(timestamps)} high-confidence segments ({refinement.confidence_level})", style="green")
                return timestamps
                
            except Exception as e:
                console.print(f"❌ Error parsing refinement JSON: {e}", style="red")
                console.print(f"Raw response: {response.text[:300]}...", style="dim yellow")
                
                # Try manual JSON extraction
                try:
                    import json
                    import re
                    
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        refinement = RefinementResult.model_validate_json(json_str)
                        timestamps = [(ts.start, ts.end) for ts in refinement.timestamps]
                        
                        console.print(f"✅ Recovered {len(timestamps)} refined segments", style="green")
                        return timestamps
                        
                except Exception as recovery_error:
                    console.print(f"❌ Refinement recovery failed: {recovery_error}", style="red")
                
                return []
                
        except Exception as e:
            console.print(f"❌ Error refining speaker with Gemini: {e}", style="red")
            return []
    
    def _get_analysis_prompt(self, target_speakers: List[str] = None, instructions: str = "") -> str:
        """Get the prompt for initial audio analysis"""
        
        # Add user instructions if provided
        instructions_prompt = ""
        if instructions:
            instructions_prompt = f"""
        USER INSTRUCTIONS: {instructions}
        
        Use these instructions to guide your analysis and speaker identification.
        """
        
        base_prompt = f"""
        # Context
        We are currently trying to train voice AI models for TV show characters and we need to extract voice segments for each character given an audio clip to produce quality training data.
        You basically need to identify the speakers in the audio and cluster their speech into segments, returning the timestamps of the segments so another processs can stitch them back together to be used for voice replication training.
        
        # Instructions
        {instructions_prompt}
        TASK: Analyze this audio and extract clean voice segments for AI training.

        For each distinct speaker you identify:
        1. Give them a descriptive name (e.g. "Speaker 1 (Deep male voice)" or use character names if you recognize them)
        2. Find clear speech segments, each segment should contain at least a short sentence or phrase

        Focus on the BEST examples of each voice for training purposes.
        """
        
        # Target speakers are now handled via general instructions
        # But keep backward compatibility for direct target_speakers parameter
        if target_speakers and not instructions:
            target_prompt = f"""
        
        FOCUS: The user wants these specific speakers: {', '.join(target_speakers)}
        
        Prioritize finding and extracting segments for these speakers.
        """
            base_prompt += target_prompt
        
        base_prompt += "\n\nThe response will be structured as JSON with speakers, descriptions, timestamps, and confidence level."
        return base_prompt
    
    def _get_refinement_prompt(self, speaker_name: str, description: str) -> str:
        """Get the prompt for speaker refinement"""
        return f"""
        This audio clip should contain the voice of {speaker_name} ({description}). 

        # Context
        We are currently trying to train voice AI models for TV show characters and we need to extract voice segments for each character given an audio clip to produce quality training data.
        You basically need to identify the speakers in the audio and cluster their speech into segments, returning the timestamps of the segments so another processs can stitch them back together to be used for voice replication training.

        # Instructions
        The audio clip has been stitched back for character {speaker_name}, yet it might still contain other speakers' voices, so again can you identify the segments that only belong to {speaker_name} again and return the timestamps of the segments so we can clean the audio sample for better training.
        """
    
    def interactive_refinement_chat(self, audio_file: str, character_name: str, description: str, 
                                   initial_timestamps: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Interactive chat with Gemini to refine audio segments based on user feedback using the stateful chat API."""
        if not self.gemini_client:
            console.print("❌ Gemini client not available", style="red")
            return initial_timestamps
        
        console.print(f"\n💬 [bold]Interactive Refinement Chat for {character_name}[/bold]", style="blue")
        console.print(f"🎵 [bold]Audio file:[/bold] [link=file://{os.path.abspath(audio_file)}]{os.path.abspath(audio_file)}[/link]", style="cyan")
        console.print("   ↑ Click the path above to listen to the audio file")
        console.print("You can now chat with Gemini to refine the audio segments!")
        console.print("Examples: 'Focus on the emotional parts', 'Remove segments with background noise', 'Keep only clear dialogue'")
        console.print("Type 'done' when satisfied, or 'restart' to go back to original timestamps.\n")
        
        current_timestamps = initial_timestamps.copy()
        
        try:
            # Convert to MP3 and upload audio file once
            mp3_file = self.audio_processor.convert_to_mp3_for_gemini(audio_file)
            audio_file_obj = self.gemini_client.files.upload(file=mp3_file)
            
            # Initialize the stateful chat session
            chat = self.gemini_client.chats.create(model="gemini-2.5-pro")
            
            # Send initial system prompt with instructions and context
            initial_prompt = f"""
            You are an expert audio editor helping me refine a list of audio timestamps for the character "{character_name}" ({description}).
            The initial timestamps are: {[(start, end) for start, end in current_timestamps]}.
            For every request I make, you MUST respond with a valid JSON object with this exact structure:
            {{
              "timestamps": [
                {{"start": "MM:SS", "end": "MM:SS"}}
              ],
              "explanation": "A brief explanation of the changes you made.",
              "confidence_level": "A confidence level like 'High', 'Medium', or 'Low'."
            }}
            I have uploaded the audio file for you to analyze. Let's begin.
            """
            
            # Start the conversation by sending the initial context. We can ignore the first response.
            console.print("🤖 Sending initial context to Gemini...", style="cyan")
            chat.send_message([initial_prompt, audio_file_obj])
            
            while True:
                # Show current segments with detailed info
                console.print(f"\n📋 [bold]Current segments for {character_name}[/bold] ({len(current_timestamps)} segments):")
                total_duration = sum(self._timestamp_to_seconds(end) - self._timestamp_to_seconds(start) for start, end in current_timestamps)
                for i, (start, end) in enumerate(current_timestamps, 1):
                    segment_duration = self._timestamp_to_seconds(end) - self._timestamp_to_seconds(start)
                    console.print(f"  {i}. {start} - {end} ({segment_duration:.1f}s)")
                
                console.print(f"📊 Total duration: {total_duration:.1f}s")
                console.print(f"🎯 Quality target: 30-60s of clear speech")
                
                # Get user input
                user_input = Prompt.ask("\n💬 Chat with Gemini (or 'done'/'restart')")
                
                if user_input.lower() == 'done':
                    if Confirm.ask("Are you satisfied with these segments?", default=True):
                        break
                    else:
                        continue
                
                if user_input.lower() == 'restart':
                    # This is tricky with a stateful chat. We will start a new chat.
                    console.print("🔄 Restarting chat with original timestamps...", style="yellow")
                    chat = self.gemini_client.chats.create(model="gemini-2.5-pro")
                    chat.send_message([initial_prompt, audio_file_obj])
                    current_timestamps = initial_timestamps.copy()
                    continue
                
                # Send user message to the chat and stream the response
                console.print("🤖 Gemini is processing your request...", style="cyan")
                response_stream = chat.send_message_stream(user_input)
                
                console.print("🔄 [bold]Gemini Response:[/bold]", style="blue")
                response_text = ""
                
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        response_text += chunk.text
                        if chunk.text.strip():
                            console.print(chunk.text, end="", style="cyan")
                console.print()  # New line
                
                # Manually parse the JSON from the accumulated text
                try:
                    # Clean up potential markdown formatting
                    if response_text.strip().startswith('```json'):
                        json_str = response_text.strip()[7:-3]
                    elif response_text.strip().startswith('```'):
                        json_str = response_text.strip()[3:-3]
                    else:
                        json_str = response_text
                    
                    # Use the Pydantic model for validation
                    refinement = RefinementResult.model_validate_json(json_str)
                    new_timestamps = [(ts.start, ts.end) for ts in refinement.timestamps]
                    
                    if new_timestamps:
                        current_timestamps = new_timestamps
                        console.print(f"\n✅ Gemini refined segments ({refinement.confidence_level})", style="green")
                        console.print(f"💡 Explanation: {refinement.explanation}", style="dim")
                    else:
                        console.print("\n⚠️ No changes made to segments", style="yellow")
                
                except (json.JSONDecodeError, Exception) as e:
                    console.print(f"\n❌ Error parsing Gemini's JSON response: {e}", style="red")
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
            3. Overall audio quality and suitability for voice training
            
            Keep it brief and conversational. Show your thinking process.
            """
            
            console.print("🌊 [bold]Gemini Streaming Preview:[/bold]", style="blue")
            
            # Use streaming for real-time response
            response_stream = self.model.generate_content(
                [prompt, audio_file_obj],
                stream=True
            )
            
            full_response = ""
            thinking_mode = False
            
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    # Handle thinking tags
                    if '<thinking>' in chunk.text:
                        thinking_mode = True
                        console.print("🧐 [dim italic]Thinking...[/dim italic]", style="yellow")
                        continue
                    elif '</thinking>' in chunk.text:
                        thinking_mode = False
                        console.print("🧐 [dim italic]Analysis:[/dim italic]", style="yellow")
                        continue
                    
                    if not thinking_mode:
                        # Print chunk immediately for streaming effect
                        console.print(chunk.text, end="", style="cyan")
                        full_response += chunk.text
            
            console.print()  # New line after streaming
            return full_response
            
        except Exception as e:
            console.print(f"❌ Error in streaming preview: {e}", style="red")
            return ""
    
