"""
CLI commands for voice trainer using Rich-Click
"""

import os
import tempfile
from typing import List, Tuple, Optional
import rich_click as click
import torch
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from pydub import AudioSegment

from ..core import VoiceTrainer
from ..utils.helpers import create_safe_filename

console = Console()

# Configure Rich-Click for beautiful help
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = "Try 'voice-trainer --help' for help."
click.rich_click.ERRORS_EPILOGUE = "To find out more, visit https://github.com/anthropics/super-voice-auto-trainer"


@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx):
    """
    🎤 **Super Voice Auto Trainer** - Multi-Video Character Voice Training
    
    A smart tool for training AI voice models from multiple YouTube videos and audio files.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@app.command()
def test_timestamps():
    """🧪 Test timestamp extraction with sample speaker data"""
    test_timestamp_extraction()


@app.command()
def test_removal():
    """🗑️ Test sound removal with sample timestamps"""
    test_sound_removal()


@app.command()
@click.argument('input_sources', nargs=-1, required=True, 
                metavar='INPUT_SOURCES...')
@click.option('--hf-token', envvar='HF_TOKEN', 
              help='🤗 Hugging Face access token - Required for pyannote workflow')
@click.option('--fish-api-key', envvar='FISH_API_KEY',
              help='🐟 Fish Audio API key - Required for model creation')
@click.option('--gemini-api-key', envvar='GEMINI_API_KEY',
              help='🤖 Gemini API key - Enables AI-powered speaker separation')
@click.option('--use-gemini', is_flag=True, default=False,
              help='🤖 Use Gemini AI for speaker separation (recommended)')
@click.option('--remove-music', is_flag=True, default=False,
              help='🎵 Remove background music before voice separation')
@click.option('--separator-model', default='UVR-MDX-NET-Inst_HQ_3.onnx',
              help='🎼 Music separation model to use')
@click.option('--skip-music-separation', is_flag=True, default=False,
              help='⏭️ Skip music separation (useful if already processed)')
def train(input_sources: tuple, hf_token: str, fish_api_key: str, gemini_api_key: str, 
          use_gemini: bool, remove_music: bool, separator_model: str, skip_music_separation: bool):
    """
    🎤 **Train voice models from multiple audio sources**
    
    **Examples:**
    
    🤖 **Gemini AI workflow (recommended):**
    ```
    voice-trainer train --use-gemini "https://youtube.com/watch?v=vid1" "video2.mp4"
    ```
    
    🔬 **Traditional pyannote workflow:**
    ```
    voice-trainer train "https://youtube.com/watch?v=vid1" "video2.mp4"
    ```
    
    🎵 **With music removal:**
    ```
    voice-trainer train --use-gemini --remove-music "movie.mp4"
    ```
    
    **INPUT_SOURCES** can be:
    - YouTube URLs (https://www.youtube.com/watch?v=...)
    - Local audio files (.wav, .mp3, .mp4, etc.)
    - Mix of both
    
    The tool processes all sources, identifies speakers, and creates character voice models.
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


def test_timestamp_extraction():
    """Test function with sample timestamp data"""
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
    from ..audio.processing import AudioProcessor
    audio_processor = AudioProcessor()
    
    # Extract speaker files to permanent location
    output_files = audio_processor.create_speaker_files_from_timestamps(audio_file, timestamp_data, output_dir)
    
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
    
    # Create audio processor
    from ..audio.processing import AudioProcessor
    audio_processor = AudioProcessor()
    
    # Remove audio segments
    console.print(f"\n🗑️ [bold]Removing {len(removal_timestamps)} audio segments...[/bold]", style="yellow")
    
    result_file = audio_processor.remove_audio_segments_by_timestamps(audio_file, removal_timestamps, output_file)
    
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