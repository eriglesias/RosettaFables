from typing import Dict, Any
from collections import Counter
import json
from pathlib import Path

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

            with open(lang_file, 'r', encoding='utf-8') as f:
                fables = json.load(f)
                self.fables_by_language[lang] = fables

                # Also organize by ID for comparative analysis
                for fable in fables:
                    fable_id = fable.get('id')
                    if fable_id:
                        if fable_id not in self.fables_by_id:
                            self.fables_by_id[fable_id] = {}
                        self.fables_by_id[fable_id][lang] = fable

    def process_all_languages(self):
        """Process fables in all languages."""
        for lang, fables in self.fables_by_language.items():
            # Get appropriate model
            nlp = get_model(lang)
            if not nlp:
                print(f"Skipping {lang} due to missing model")
                continue
    
            # Process each fable
            processed_fables = []
            for fable in fables:
                processed = preprocess_fable(fable, nlp)
                processed_fables.append(processed)

            # Save processed results
            output_file = self.data_dir.parent / "analysis" / f"processed_{lang}.json"
            output_file.parent.mkdir(exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_fables, f, ensure_ascii=False, indent=2)

            print(f"Processed {len(processed_fables)} fables in {lang}")
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