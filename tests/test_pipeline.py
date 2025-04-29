from pathlib import Path
import logging
from aesop_spacy.pipeline.pipeline import FablePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_pipeline")

# Set up directories
data_dir = Path("data")
output_dir = Path("output")

# Initialize pipeline
pipeline = FablePipeline(data_dir, output_dir)

# Test specific analyzer
test_results = pipeline.analyze(analysis_types=['clustering'])
logger.info(f"Completed analysis with {len(test_results)} results")