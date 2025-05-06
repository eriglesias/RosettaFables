#log_utils.py
"""
Logging utilities for the Aesop fables analysis project.

This module provides formatting utilities to create visually distinct
and well-organized log outputs for different processing phases.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any, Optional


def section_header(name: str, width: int = 80, logger: Optional[logging.Logger] = None) -> str:
    """
    Create a section header with the given name.
    
    Args:
        name: Name of the section
        width: Width of the header (default: 80)
        logger: Logger object (unused, kept for backward compatibility)
        
    Returns:
        Formatted header string
    """
    # This returns a complete header as a single string to be logged as one message
    header = (
        f"\n"
        f"{'=' * width}\n"
        f"{name.center(width)}\n"
        f"{'=' * width}"
    )
    return header

def subsection_header(name: str, width: int = 80, logger: Optional[logging.Logger] = None) -> str:
    """
    Create a subsection header with the given name.
    
    Args:
        name: Name of the subsection
        width: Width of the header (default: 80)
        logger: Logger object (unused, kept for backward compatibility)
        
    Returns:
        Formatted header string
    """
    # This returns a complete header as a single string to be logged as one message
    header = (
        f"\n"
        f"{'-' * width}\n"
        f"{name.center(width)}\n"
        f"{'-' * width}"
    )
    return header

def log_timing(func: Callable) -> Callable:
    """
    Decorator to log the time taken by a function.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function that logs timing information
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info("Starting %s...", func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info("Completed %s in %.2f seconds", func.__name__, duration)
        return result
    return wrapper

def format_count(name: str, count: int) -> str:
    """Format a count with the appropriate singular/plural form."""
    return f"{count} {name}{'s' if count != 1 else ''}"

def format_percentage(name: str, value: float) -> str:
    """Format a percentage value with descriptive label."""
    return f"{name}: {value:.2f}%"

def wrap_analysis_result(result: Any, name: str, logger: logging.Logger) -> Any:
    """Wrap an analysis result with success/failure logging."""
    if isinstance(result, dict) and 'error' in result:
        logger.error("%s analysis failed: %s", name, result['error'])
    else:
        logger.info("%s analysis completed successfully", name)
    return result
