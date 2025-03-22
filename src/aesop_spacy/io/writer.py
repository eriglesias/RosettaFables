# src/aesop_spacy/io/writer.py
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging


class OutputWriter:
    """Responsible for writing processed data to output files."""

    def __init__(self, output_dir: Path):
        """
        Initialize the writer.
        
        Args:
            output_dir: Directory to write output files
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure that the necessary output directories exist."""
        # Create main output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories if they don't exist
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)

    def save_processed_fables(self, fables: List[Dict[str, Any]], language: str) -> Path:
        """
        Save processed fables to a JSON file.
        
        Args:
            fables: List of processed fable dictionaries
            language: Language code for the fables
            
        Returns:
            Path to the saved file
        """
        if not fables:
            self.logger.warning(f"No fables to save for language '{language}'")
            return None
        
        # Determine the output file path
        output_file = self.output_dir / "processed" / f"fables_{language}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(fables, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved {len(fables)} processed fables to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving processed fables for {language}: {e}")
            return None
    
    def save_analysis_results(self, analysis_data: Dict[str, Any], language: str, 
                              analysis_type: str) -> Path:
        """
        Save analysis results to a JSON file.
        
        Args:
            analysis_data: Analysis results dictionary
            language: Language code for the analysis
            analysis_type: Type of analysis (e.g., 'pos', 'entity')
            
        Returns:
            Path to the saved file
        """
        # Determine the output file path
        output_file = self.output_dir / "analysis" / f"{analysis_type}_{language}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved {analysis_type} analysis for {language} to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving {analysis_type} analysis for {language}: {e}")
            return None
    
    def save_comparison_results(self, comparison_data: Dict[str, Any], 
                               comparison_id: str) -> Path:
        """
        Save fable comparison results to a JSON file.
        
        Args:
            comparison_data: Comparison results dictionary
            comparison_id: Identifier for the comparison (e.g., fable ID)
            
        Returns:
            Path to the saved file
        """
        # Determine the output file path
        output_file = self.output_dir / "analysis" / f"comparison_{comparison_id}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved comparison results for fable {comparison_id} to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving comparison for fable {comparison_id}: {e}")
            return None