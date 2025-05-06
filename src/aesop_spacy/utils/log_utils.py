"""
Logging utilities for the Aesop fables analysis project.

This module provides formatting utilities to create visually distinct
and well-organized log outputs for different processing phases.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any, Optional

# ANSI color codes for terminal output
COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'END': '\033[0m'
}

def section_header(name: str, width: int = 80, logger: Optional[logging.Logger] = None) -> None:
    """
    Print a section header with the given name.
    
    Args:
        name: Name of the section
        width: Width of the header (default: 80)
        logger: Logger to use (if None, prints to stdout)
    """
    header = f"\n{COLORS['BOLD']}{COLORS['BLUE']}{'=' * width}{COLORS['END']}"
    title = f"{COLORS['BOLD']}{COLORS['BLUE']}{name.center(width)}{COLORS['END']}"
    footer = f"{COLORS['BOLD']}{COLORS['BLUE']}{'=' * width}{COLORS['END']}\n"
    
    if logger:
        logger.info(header)
        logger.info(title)
        logger.info(footer)
    else:
        print(header)
        print(title)
        print(footer)

def subsection_header(name: str, width: int = 80, logger: Optional[logging.Logger] = None) -> None:
    """
    Print a subsection header with the given name.
    
    Args:
        name: Name of the subsection
        width: Width of the header (default: 80)
        logger: Logger to use (if None, prints to stdout)
    """
    header = f"\n{COLORS['BOLD']}{COLORS['GREEN']}{'-' * width}{COLORS['END']}"
    title = f"{COLORS['BOLD']}{COLORS['GREEN']}{name.center(width)}{COLORS['END']}"
    footer = f"{COLORS['BOLD']}{COLORS['GREEN']}{'-' * width}{COLORS['END']}\n"
    
    if logger:
        logger.info(header)
        logger.info(title)
        logger.info(footer)
    else:
        print(header)
        print(title)
        print(footer)

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
        logger.info(f"Starting {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
        return result
    return wrapper

def format_count(name: str, count: int) -> str:
    """Format a count for logging."""
    return f"{COLORS['BOLD']}{count}{COLORS['END']} {name}{'s' if count != 1 else ''}"

def format_percentage(name: str, value: float) -> str:
    """Format a percentage for logging."""
    color = COLORS['GREEN'] if value > 75 else COLORS['YELLOW'] if value > 50 else COLORS['RED']
    return f"{name}: {color}{value:.2f}%{COLORS['END']}"

def wrap_analysis_result(result: Any, name: str, logger: logging.Logger) -> Any:
    """Wrap an analysis result with success/failure logging."""
    if isinstance(result, dict) and 'error' in result:
        logger.error(f"{name} analysis failed: {result['error']}")
    else:
        logger.info(f"{name} analysis completed successfully")
    return result