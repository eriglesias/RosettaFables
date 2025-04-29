#main.py
"""
Aesop Fable Analysis Tool - Main Entry Point

This module serves as the main entry point for processing and analyzing multilingual
Aesop fables using NLP techniques. It orchestrates the pipeline components to extract,
process, and analyze fable texts across different languages.

Features:
- Processes markdown-formatted fable files into structured JSON
- Analyzes linguistic features across languages and fables
- Generates comparative analysis between different language versions
- Provides command-line configuration options

Usage:
    python main.py [OPTIONS]

Options:
    --data-dir PATH       Path to data directory containing fables
    --output-dir PATH     Path to save processed and analyzed data
    --only-process        Only process raw fables to JSON (skip analysis)
    --only-analyze        Only analyze previously processed fables
    --debug               Enable detailed debug logging

Example:
    python main.py --data-dir ./my_fables --output-dir ./results --debug


Dependencies:
    - pathlib: For cross-platform path handling
    - logging: For application logging
    - argparse: For command-line argument parsing
    - json: For reading/writing structured data
"""

import logging
import argparse
import json
import sys
from pathlib import Path
from aesop_spacy.pipeline.pipeline import FablePipeline

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(description="Process and analyze Aesop's fables")

    parser.add_argument('--data-dir', type=str, help='Data directory path')
    parser.add_argument('--output-dir', type=str, help='Output directory path')
    parser.add_argument('--only-process', action='store_true', help='Only process fables')
    parser.add_argument('--only-analyze', action='store_true', help='Only analyze processed fables')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser.parse_args()


def setup_logging(debug_mode=False):
    """Configure application logging based on verbosity level"""
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    return logging.getLogger("main")


def setup_directories(data_dir, output_dir, logger):
    """Ensure required directories exist and return validated paths"""
    # Create absolute paths if relative paths provided
    data_dir = Path(data_dir) if data_dir else project_root / "data"
    output_dir = Path(output_dir) if output_dir else project_root / "data"

    logger.info("Using data directory: %s", data_dir)
    logger.info("Using output directory: %s", output_dir)

    # Ensure directories exist
    for subdir in ["raw/fables", "processed", "analysis"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Verify fable source file
    fable_md_path = data_dir / "raw" / "fables" / "initial_fables.md"
    logger.info("Looking for fables at: %s (exists: %s)", 
              fable_md_path, fable_md_path.exists())

    # Check processed files
    json_files = list((data_dir / "processed").glob("*.json"))
    logger.info("Found %d JSON files in processed directory", len(json_files))

    return data_dir, output_dir


def print_analysis_summary(logger, results):
    """Print a summary of the analysis results to the console"""
    if 'pos_distribution' in results:
        for lang, pos_dist in results['pos_distribution'].items():
            logger.info("\nPOS distribution for %s:", lang)
            for pos, percentage in list(pos_dist.items())[:5]:  # Top 5
                logger.info("  %s: %.2f%%", pos, percentage)

    if 'fable_comparisons' in results:
        for fable_id, comparison in list(results['fable_comparisons'].items())[:3]:
            logger.info("\nFable %s comparison across %d languages:", 
                       fable_id, len(comparison['languages']))
            logger.info("  Languages: %s", ', '.join(comparison['languages']))
            
            # Safely access token_counts
            if 'token_counts' in comparison:
                logger.info("  Token counts: %s", comparison['token_counts'])
            else:
                logger.info("  Token counts: Not available")


def save_analysis_summary(output_dir, results):
    """Save analysis summary to JSON file"""
    summary_file = output_dir / "analysis" / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'languages_processed': list(results.get('pos_distribution', {}).keys()),
            'fables_compared': list(results.get('fable_comparisons', {}).keys()),
            'analysis_types': list(results.keys())
        }, f, indent=2)


def run_pipeline(args, data_dir, output_dir, logger):
    """Run the fable processing pipeline based on command line arguments"""
    pipeline = FablePipeline(data_dir, output_dir)

    if args.only_analyze:
        logger.info("Running analysis only")
        return pipeline.analyze(), False

    elif args.only_process:
        logger.info("Running processing only")
        pipeline.run()
        return None, False

    else:
        logger.info("Running full pipeline")
        pipeline.run()
        results = pipeline.analyze()
        return results, True


def main():
    """Main entry point for Aesop fable analysis"""
    # Parse arguments and set up logging
    args = parse_arguments()
    logger = setup_logging(args.debug)

    try:
        # Set up directories
        data_dir, output_dir = setup_directories(args.data_dir, args.output_dir, logger)

        # Run the pipeline
        analysis_results, save_results = run_pipeline(args, data_dir, output_dir, logger)

        # Handle results if present
        if analysis_results:
            print_analysis_summary(logger, analysis_results)

            if save_results:
                save_analysis_summary(output_dir, analysis_results)

        logger.info("Aesop fable processing complete")

    except FileNotFoundError as e:
        logger.error("Required file not found: %s", e)
    except ImportError as e:
        logger.error("Module import error: %s", e)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON data: %s", e)
    except ValueError as e:
        logger.error("Data format error: %s", e)
    except PermissionError as e:
        logger.error("Permission denied accessing files: %s", e)
    except OSError as e:
        logger.error("OS error occurred: %s", e)
    except Exception as e:  # Fallback for unexpected errors
        logger.error("Unexpected error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()