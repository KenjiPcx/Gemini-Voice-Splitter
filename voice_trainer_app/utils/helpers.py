"""
Helper utility functions
"""

import re


def create_safe_filename(name: str) -> str:
    """Create a safe filename from a string by removing problematic characters"""
    # Replace spaces with underscores
    safe_name = name.replace(' ', '_')
    
    # Remove or replace problematic characters
    safe_name = re.sub(r'[()<>,\'\"\\/:*?|]', '', safe_name)
    
    # Replace multiple underscores with single ones
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    return safe_name