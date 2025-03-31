import json
from collections import Counter
from pathlib import Path
import logging
from typing import Dict, List, Any

class EntityAnalyzer:
    """Analyze entities in fables across languages."""
    
    def __init__(self, analysis_dir: Path):
        """Initialize analyzer with the directory containing processed fables."""
        self.analysis_dir = analysis_dir
        self.logger = logging.getLogger(__name__)
    
    def analyze_entity_distribution(self, language: str) -> Dict[str, Any]:
        """
        Analyze entity type distribution for a specific language.
        
        Args:
            language: ISO language code
            
        Returns:
            Dictionary with entity type statistics
        """
        processed_file = self.analysis_dir / f"processed_{language}.json"
        if not processed_file.exists():
            self.logger.warning(f"No processed file found for {language}")
            return {}
        
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                fables = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading processed file for {language}: {e}")
            return {}
        
        # Count entities by type
        entity_counter = Counter()
        total_entities = 0
        
        # Store examples of each entity type
        entity_examples = {}
        
        for fable in fables:
            entities = fable.get('entities', [])
            for entity_data in entities:
                # Handle both tuple format and dictionary format
                if isinstance(entity_data, tuple) and len(entity_data) >= 2:
                    entity_text, entity_type = entity_data[0], entity_data[1]
                elif isinstance(entity_data, dict):
                    entity_text = entity_data.get('text', '')
                    entity_type = entity_data.get('label', '')
                else:
                    continue
                
                if entity_type:
                    entity_counter[entity_type] += 1
                    total_entities += 1
                    
                    # Store examples of each entity type
                    if entity_type not in entity_examples:
                        entity_examples[entity_type] = []
                    
                    if entity_text and len(entity_examples[entity_type]) < 2:
                        entity_examples[entity_type].append(entity_text)
        
        # Convert to percentages and add examples
        result = {}
        for entity_type, count in entity_counter.most_common():
            result[entity_type] = {
                'percentage': count / total_entities * 100 if total_entities > 0 else 0,
                'count': count,
                'examples': entity_examples.get(entity_type, [])
            }
        
        # Save results to a file
        output_file = self.analysis_dir / f"entity_analysis_{language}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved entity analysis for {language} to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving entity analysis for {language}: {e}")
        
        return result