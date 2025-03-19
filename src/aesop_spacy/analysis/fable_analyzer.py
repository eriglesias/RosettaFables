from typing import Dict, Any
from collections import Counter
import json
from pathlib import Path
from spacy.tokens import Doc, Span, Token
import numpy as np
from ..models.model_manager import get_model
from ..preprocessing.text_processor import preprocess_fable

class FableAnalyzer:
    """Analyze fables across different languages."""
    def __init__(self, data_dir: Path):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing processed fable JSON files
        """
        self.data_dir = data_dir
        self.fables_by_language = {}
        self.fables_by_id = {}
        # Load fables
        self._load_fables()

    def _load_fables(self):
        """Load fables from JSON files."""
        # Load by language
        for lang_file in self.data_dir.glob("fables_*.json"):
            lang = lang_file.stem.split('_')[1]
            
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    # Parse the JSON file
                    fables = json.load(f)
                    
                    # Debug print to see what we're getting
                    print(f"Loaded {len(fables)} entries from {lang_file.name}")
                    print(f"Type of first entry: {type(fables[0])}")
                    
                    # Ensure we're working with a list of dictionaries
                    if isinstance(fables, list):
                        self.fables_by_language[lang] = fables
                        
                        # Process each fable dictionary
                        for fable in fables:
                            if isinstance(fable, dict):  # Make sure it's a dictionary
                                fable_id = fable.get('id')
                                if fable_id:
                                    if fable_id not in self.fables_by_id:
                                        self.fables_by_id[fable_id] = {}
                                    self.fables_by_id[fable_id][lang] = fable
                            else:
                                print(f"Warning: Expected dict but got {type(fable)}: {fable}")
                    else:
                        print(f"Warning: Expected list but got {type(fables)}")
            except Exception as e:
                print(f"Error loading {lang_file.name}: {e}")

    def serialize_spacy_objects(self, obj):
        """
        Recursively convert spaCy objects to serializable Python types.
        
        Args:
            obj: Any object or data structure
            
        Returns:
            JSON-serializable version of the object
        """
        # Handle different types of objects
        if isinstance(obj, (Doc, Span)):
            # Convert Doc or Span to a dictionary with basic properties
            return {
                "text": obj.text,
                "start": getattr(obj, "start", 0),
                "end": getattr(obj, "end", len(obj)),
                "label_": getattr(obj, "label_", ""),
                # Add any other attributes you need
            }
        elif isinstance(obj, Token):
            # Convert Token to a dictionary
            return {
                "text": obj.text,
                "pos_": obj.pos_,
                "tag_": obj.tag_,
                "dep_": obj.dep_,
                "lemma_": obj.lemma_,
                "is_stop": obj.is_stop,
                # Add any other attributes you need
            }
        elif isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists
            return obj.tolist()
        elif isinstance(obj, dict):
            # Recursively process dictionary values
            return {key: self.serialize_spacy_objects(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list items
            return [self.serialize_spacy_objects(item) for item in obj]
        elif isinstance(obj, tuple):
            # Convert tuples to lists and recursively process
            return [self.serialize_spacy_objects(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Handle custom objects by converting them to dictionaries
            return self.serialize_spacy_objects(obj.__dict__)
        else:
            # Return other types as-is (assumes they're JSON-serializable)
            return obj

    def process_all_languages(self):
        """Process fables in all languages."""
        for lang, fables in self.fables_by_language.items():
            # Get appropriate model
            nlp = get_model(lang)
            if not nlp:
                print(f"Skipping {lang} due to missing model")
                continue
            print(f"Processing {len(fables)} fables in language: {lang}")
            # Process each fable
            processed_fables = []
            for fable in fables:
                try: 
                    processed = preprocess_fable(fable, nlp)
                    processed_fables.append(processed)
                except Exception as e:
                    print(f"Error processing fable {fable.get('id', 'unknown')}: {e}")

            # Only save if we processed any fables
            if processed_fables:
                # Convert spaCy objects to serializable types
                serializable_fables = self.serialize_spacy_objects(processed_fables)
                
                # Save processed results
                output_file = self.data_dir.parent / "analysis" / f"processed_{lang}.json"
                output_file.parent.mkdir(exist_ok=True, parents=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_fables, f, ensure_ascii=False, indent=2)
                    
                print(f"✅ Processed {len(processed_fables)} fables in {lang}")


    def analyze_pos_distribution(self, language: str) -> Dict[str, float]:
        """
        Analyze part-of-speech distribution for a language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with POS tag frequencies
        """
        pos_counts = Counter()
        total_tokens = 0
        
        # Load processed fables
        processed_file = self.data_dir.parent / "analysis" / f"processed_{language}.json"
        if not processed_file.exists():
            print(f"No processed data for {language}")
            return {}

        with open(processed_file, 'r', encoding='utf-8') as f:
            fables = json.load(f)

        # Count POS tags
        for fable in fables:
            for _, pos in fable.get('pos_tags', []):
                pos_counts[pos] += 1
                total_tokens += 1

        # Convert to percentages
        return {pos: count/total_tokens*100 for pos, count in pos_counts.items()}

    def compare_fable_across_languages(self, fable_id: str) -> Dict[str, Any]:
        """
        Compare the same fable across different languages.
        
        Args:
            fable_id: Fable ID to compare
            
        Returns:
            Comparison data
        """
        if fable_id not in self.fables_by_id:
            return {"error": f"Fable ID {fable_id} not found"}

        languages = list(self.fables_by_id[fable_id].keys())
        comparison = {
            "languages": languages,
            "token_counts": {},
            "entity_counts": {},
            "has_moral": {}
        }

        # Load processed data for each language
        for lang in languages:
            processed_file = self.data_dir.parent / "analysis" / f"processed_{lang}.json"
            if not processed_file.exists():
                continue

            with open(processed_file, 'r', encoding='utf-8') as f:
                fables = json.load(f)

            # Find the specific fable
            for fable in fables:
                if fable.get('id') == fable_id:
                    comparison['token_counts'][lang] = len(fable.get('tokens', []))
                    comparison['entity_counts'][lang] = len(fable.get('entities', []))
                    comparison['has_moral'][lang] = bool(fable.get('moral', {}).get('text', ''))
                    break
    
        return comparison