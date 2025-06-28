#!/usr/bin/env python3
"""
Super Voice Auto Trainer - Main entry point
"""

from dotenv import load_dotenv
from voice_trainer_app.cli import app

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    app()