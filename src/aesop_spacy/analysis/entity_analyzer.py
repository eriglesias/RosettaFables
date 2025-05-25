"""
Entity analysis module for Aesop's fables with clean directory structure.

This module analyzes entity distributions across languages with a defensive approach
to file loading and clear separation of concerns.
"""

import json
from collections import Counter
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional


class EntityAnalyzer:
    """
    Analyze entities in processed fables across languages.
    
    This class follows a defensive programming approach and maintains
    clear separation between input and output directories.
    """
    
    def __init__(self, analysis_dir: Path):
        """
        Initialize analyzer with clean directory structure.
        
        The directory structure should be:
        - data/data_handled/processed/     (where processed fables live)
        - data/data_handled/analysis/      (where analysis results go)
        
        Args:
            analysis_dir: Analysis output directory (data_handled/analysis)
        """
        self.analysis_dir = Path(analysis_dir).resolve()
        self.logger = logging.getLogger(__name__)
        
        # The processed files should be in the same data_handled tree
        # This creates a clean separation: data_handled contains ALL processed outputs
        self.processed_dir = self._find_processed_directory()
        
        # Ensure our output directory exists
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug("EntityAnalyzer initialized:")
        self.logger.debug("  Analysis dir: %s", self.analysis_dir)
        self.logger.debug("  Processed dir: %s", self.processed_dir)
    
    def analyze_entity_distribution(self, language: str) -> Dict[str, Any]:
        """
        Analyze entity type distribution for a specific language.
        
        Uses defensive programming to handle missing files gracefully.
        
        Args:
            language: ISO language code (e.g., 'en', 'de', 'nl')
            
        Returns:
            Dictionary with entity statistics or empty dict if no data found
        """
        processed_file = self._locate_processed_file(language)
        
        if not processed_file:
            self.logger.warning("No processed file found for language '%s'", language)
            return {}
        
        fables_data = self._load_fables_safely(processed_file, language)
        if not fables_data:
            return {}
        
        return self._analyze_entities(fables_data, language)
    
    def _locate_processed_file(self, language: str) -> Optional[Path]:
        """
        Locate the processed file for a language using defensive search.
        
        This method implements a fallback search strategy in case the
        directory structure has variations.
        
        Args:
            language: ISO language code
            
        Returns:
            Path to processed file or None if not found
        """
        # Primary location (the correct one)
        primary_path = self.processed_dir / f"fables_{language}.json"
        
        if primary_path.exists():
            self.logger.debug("Found processed file at primary location: %s", primary_path)
            return primary_path
        
        # Fallback locations for robustness
        fallback_paths = [
            # Legacy location (in case old structure exists)
            self.analysis_dir.parent.parent / "data_raw" / "processed" / f"fables_{language}.json",
            # Alternative naming conventions
            self.processed_dir / f"processed_{language}.json",
            self.analysis_dir / f"fables_{language}.json",
        ]
        
        for path in fallback_paths:
            if path.exists():
                self.logger.info("Found processed file at fallback location: %s", path)
                return path
        
        # Log search locations for debugging
        all_paths = [primary_path] + fallback_paths
        self.logger.debug("Searched for processed file in these locations:")
        for path in all_paths:
            self.logger.debug("  %s (exists: %s)", path, path.exists())
        
        return None
    
    def _load_fables_safely(self, file_path: Path, language: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load fables data with comprehensive error handling.
        
        Args:
            file_path: Path to the processed file
            language: Language code for error logging
            
        Returns:
            List of fables or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                self.logger.error("Expected list of fables for %s, got %s", language, type(data))
                return None
            
            # Validate we have actual fables
            if not data:
                self.logger.warning("Empty fables list for language %s", language)
                return None
            
            self.logger.debug("Loaded %d fables for %s", len(data), language)
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON in processed file for %s: %s", language, e)
        except FileNotFoundError:
            self.logger.error("Processed file not found: %s", file_path)
        except PermissionError:
            self.logger.error("Permission denied reading file: %s", file_path)
        except Exception as e:
            self.logger.error("Unexpected error loading processed file for %s: %s", language, e)
        
        return None
    
    def _analyze_entities(self, fables: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        """
        Extract and analyze entity statistics from fables.
        
        This method processes entity data and builds comprehensive statistics
        while handling various data formats defensively.
        
        Args:
            fables: List of processed fable dictionaries
            language: Language code for context
            
        Returns:
            Dictionary with entity type statistics
        """
        entity_counter = Counter()
        entity_examples = {}
        total_entities = 0
        processed_fables = 0
        
        for fable in fables:
            if not isinstance(fable, dict):
                self.logger.debug("Skipping non-dictionary fable in %s", language)
                continue
            
            processed_fables += 1
            entities = fable.get('entities', [])
            
            for entity_data in entities:
                entity_info = self._parse_entity_safely(entity_data)
                if not entity_info:
                    continue
                
                entity_text, entity_type = entity_info
                
                # Count this entity
                entity_counter[entity_type] += 1
                total_entities += 1
                
                # Store examples (limited to prevent memory bloat)
                self._store_entity_example(entity_type, entity_text, entity_examples)
        
        # Build comprehensive result
        result = self._build_result_dict(entity_counter, entity_examples, total_entities, processed_fables, language)
        
        # Save results for future use
        self._save_analysis_results(result, language)
        
        return result

    def _find_processed_directory(self) -> Path:
        """
        Bulletproof method to find the processed directory.
        Uses multiple fallback strategies.
        """
        # Strategy 1: Standard sibling directory
        candidate = self.analysis_dir.parent / "processed"
        if candidate.exists():
            self.logger.debug("Found processed dir via sibling: %s", candidate)
            return candidate
        
        # Strategy 2: Look for data_handled structure anywhere up the tree
        current = self.analysis_dir
        for _ in range(5):  # Reasonable search depth
            data_handled = current / "data_handled" / "processed"
            if data_handled.exists():
                self.logger.debug("Found processed dir via data_handled: %s", data_handled)
                return data_handled
            current = current.parent
        
        # Strategy 3: Direct search from project root
        # This handles cases where the directory structure varies
        project_patterns = [
            "data/data_handled/processed",
            "data_handled/processed", 
            "processed"
        ]
        
        for pattern in project_patterns:
            candidate = self.analysis_dir
            # Walk up to find project root, then apply pattern
            for _ in range(5):
                test_path = candidate / pattern
                if test_path.exists():
                    self.logger.debug("Found processed dir via pattern %s: %s", pattern, test_path)
                    return test_path
                candidate = candidate.parent
        
        # Fallback: Create the expected location
        fallback = self.analysis_dir.parent / "processed"
        self.logger.warning("Could not find processed directory, using fallback: %s", fallback)
        return fallback


    def _parse_entity_safely(self, entity_data: Any) -> Optional[tuple]:
        """
        Parse entity data from various formats with defensive handling.
        
        Handles multiple data formats that might come from different
        processing pipelines or spaCy versions.
        
        Args:
            entity_data: Entity data in various formats
            
        Returns:
            Tuple of (entity_text, entity_type) or None if invalid
        """
        try:
            # Format 1: Tuple/List (text, type, start, end)
            if isinstance(entity_data, (tuple, list)) and len(entity_data) >= 2:
                entity_text = str(entity_data[0]).strip()
                entity_type = str(entity_data[1]).strip()
                
            # Format 2: Dictionary {'text': ..., 'label': ...}
            elif isinstance(entity_data, dict):
                entity_text = str(entity_data.get('text', entity_data.get('entity', ''))).strip()
                entity_type = str(entity_data.get('label', entity_data.get('type', ''))).strip()
                
            # Format 3: String representation (fallback)
            elif isinstance(entity_data, str):
                # This might be a string representation of an entity
                # Skip for now as it's ambiguous
                return None
                
            else:
                self.logger.debug("Unknown entity data format: %s", type(entity_data))
                return None
            
            # Validate we have meaningful data
            if entity_text and entity_type and len(entity_text.strip()) > 0:
                return (entity_text, entity_type)
                
        except (AttributeError, KeyError, TypeError) as e:
            self.logger.debug("Error parsing entity data %s: %s", entity_data, e)
        
        return None
    
    def _store_entity_example(self, entity_type: str, entity_text: str, examples_dict: Dict[str, List[str]]) -> None:
        """
        Store entity examples with deduplication and limits.
        
        Args:
            entity_type: Type of entity
            entity_text: Text of the entity
            examples_dict: Dictionary to store examples in
        """
        if entity_type not in examples_dict:
            examples_dict[entity_type] = []
        
        # Limit examples to prevent memory bloat and avoid duplicates
        if len(examples_dict[entity_type]) < 5 and entity_text not in examples_dict[entity_type]:
            examples_dict[entity_type].append(entity_text)
    
    def _build_result_dict(self, counter: Counter, examples: Dict[str, List[str]], 
                           total_entities: int, processed_fables: int, language: str) -> Dict[str, Any]:
        """
        Build the final result dictionary with comprehensive statistics.
        
        Args:
            counter: Counter of entity types
            examples: Dictionary of entity examples
            total_entities: Total number of entities found
            processed_fables: Number of fables processed
            language: Language code
            
        Returns:
            Comprehensive result dictionary
        """
        result = {}
        
        # Add entity type statistics
        for entity_type, count in counter.most_common():
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            
            result[entity_type] = {
                'count': count,
                'percentage': round(percentage, 2),
                'examples': examples.get(entity_type, [])
            }
        
        # Add metadata for context
        result['_metadata'] = {
            'language': language,
            'total_entities': total_entities,
            'processed_fables': processed_fables,
            'unique_entity_types': len(counter),
            'analysis_timestamp': self._get_timestamp()
        }
        
        self.logger.info("Entity analysis for %s: %d entities across %d types from %d fables", 
                        language, total_entities, len(counter), processed_fables)
        
        return result
    
    def _save_analysis_results(self, result: Dict[str, Any], language: str) -> None:
        """
        Save entity analysis results to file with error handling.
        
        Args:
            result: Analysis results dictionary
            language: Language code
        """
        output_file = self.analysis_dir / f"entity_{language}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Saved entity analysis for %s to %s", language, output_file)
            
        except PermissionError:
            self.logger.error("Permission denied writing to %s", output_file)
        except OSError as e:
            self.logger.error("OS error saving entity analysis for %s: %s", language, e)
        except Exception as e:
            self.logger.error("Unexpected error saving entity analysis for %s: %s", language, e)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()