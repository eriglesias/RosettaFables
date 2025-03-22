# main.py
import logging
import argparse
import json
import sys
from pathlib import Path

# Set up a proper Python path to allow absolute imports
# Get the project root (parent directory of src)
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent  # Assuming main.py is in src/aesop_spacy/
project_root = src_dir.parent         # Get the parent of src/
sys.path.insert(0, str(project_root)) # Add project root to Python path


# Now we can use absolute imports
from aesop_spacy.pipeline.pipeline import FablePipeline
from aesop_spacy.models.model_manager import get_model  # Import this directly too


def main():
    """
    Main entry point for the Aesop fable analysis project.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("main")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process and analyze Aesop's fables")
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, help='Path to output directory')
    parser.add_argument('--only-process', action='store_true', help='Only process fables without analysis')
    parser.add_argument('--only-analyze', action='store_true', help='Only analyze previously processed fables')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Set up directories correctly
    # Use command line arguments if provided, otherwise use defaults
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "data"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "data"
    
    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Using output directory: {output_dir}")
    
    # Debug: Check important data files
    fable_md_path = data_dir / "raw" / "fables" / "initial_fables.md"
    logger.info(f"Looking for initial_fables.md at: {fable_md_path}")
    logger.info(f"File exists: {fable_md_path.exists()}")
    
    # Create directories if they don't exist
    for subdir in ["raw", "processed", "analysis"]:
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            logger.info(f"Creating directory: {subdir_path}")
            subdir_path.mkdir(parents=True, exist_ok=True)
    
    # Create fables subdir if it doesn't exist
    fables_dir = data_dir / "raw" / "fables"
    if not fables_dir.exists():
        logger.info(f"Creating fables directory: {fables_dir}")
        fables_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug: Check if any JSON files exist in processed dir
    json_files = list((data_dir / "processed").glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in processed directory: {[f.name for f in json_files]}")
    
    # Initialize the pipeline
    try:
        pipeline = FablePipeline(data_dir, output_dir)
        
        # Run the appropriate pipeline steps
        if args.only_analyze:
            # Only run analysis on previously processed fables
            logger.info("Running analysis only")
            analysis_results = pipeline.analyze()
            
            # Print some sample results
            if 'pos_distribution' in analysis_results:
                for lang, pos_dist in analysis_results['pos_distribution'].items():
                    logger.info(f"\nPOS distribution for {lang}:")
                    for pos, percentage in list(pos_dist.items())[:5]:  # Show top 5
                        logger.info(f"  {pos}: {percentage:.2f}%")
                        
            if 'fable_comparisons' in analysis_results:
                for fable_id, comparison in list(analysis_results['fable_comparisons'].items())[:3]:  # Show first 3
                    logger.info(f"\nFable {fable_id} comparison across {len(comparison['languages'])} languages:")
                    logger.info(f"  Languages: {', '.join(comparison['languages'])}")
                    logger.info(f"  Token counts: {comparison['token_counts']}")
        
        elif args.only_process:
            # Only process fables without analysis
            logger.info("Running processing only")
            pipeline.run()
            
        else:
            # Run the full pipeline
            logger.info("Running full pipeline (processing + analysis)")
            pipeline.run()
            analysis_results = pipeline.analyze()
            
            # Save overall analysis summary
            try:
                summary_file = output_dir / "analysis" / "summary.json"
                summary_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'languages_processed': list(analysis_results.get('pos_distribution', {}).keys()),
                        'fables_compared': list(analysis_results.get('fable_comparisons', {}).keys()),
                        'analysis_types': list(analysis_results.keys())
                    }, f, indent=2)
                    
                logger.info(f"Analysis summary saved to {summary_file}")
            except Exception as e:
                logger.error(f"Error saving analysis summary: {e}")
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    
    logger.info("Aesop fable processing and analysis complete")


# This allows the script to be run directly
if __name__ == "__main__":
    main()