from pathlib import Path
import logging
import json
from aesop_spacy.pipeline.pipeline import FablePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_pipeline")

# Set up directories
data_dir = Path("data")
output_dir = Path("output")

def test_analyzer(pipeline, analysis_type):
    """Test a specific analyzer and print basic results info"""
    logger.info(f"Testing {analysis_type} analyzer...")
    results = pipeline.analyze(analysis_types=[analysis_type])
    
    if analysis_type in results:
        logger.info(f"{analysis_type} analysis completed successfully")
        # Save a sample of results for inspection
        sample_file = output_dir / f"test_{analysis_type}_sample.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(results[analysis_type], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved sample results to {sample_file}")
    else:
        logger.error(f"{analysis_type} analysis failed - no results returned")
    
    return results

def main():
    # Initialize pipeline
    pipeline = FablePipeline(data_dir, output_dir)
    
    # Ensure we have processed data
    pipeline.run(use_processed=True)
    
    # Test each analyzer separately
    analysis_types = [
        'pos', 'entity', 'character', 'comparison', 'moral',  # Basic types
        'clustering', 'sentiment', 'style', 'syntax',          # Advanced types
        'nlp_techniques', 'stats', 'cross_language'            # Additional types
    ]
    
    for analysis_type in analysis_types:
        test_analyzer(pipeline, analysis_type)
    
    # Finally test all analyzers together
    logger.info("Testing all analyzers together...")
    all_results = pipeline.analyze()
    logger.info(f"All analyses completed with {len(all_results)} result types")

if __name__ == "__main__":
    main()