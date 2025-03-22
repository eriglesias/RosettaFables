from typing import Dict, Any
from collections import Counter
import json
import re
import logging
from pathlib import Path
from spacy.tokens import  Token
from ..models.model_manager import get_model
from ..preprocessing.processor import preprocess_fable

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
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s')
        # Load fables
        self._load_fables()
        # Fix empty language codes
        self._fix_language_codes()

    def _load_fables(self):
        """Load fables from JSON files."""
        # Load all fable files, not just those with language codes
        for lang_file in self.data_dir.glob("fables*.json"):
            # Extract language from filename
            parts = lang_file.stem.split('_')
            
            # Handle different filename patterns
            if len(parts) > 1:
                lang = parts[1]  # Normal case: "fables_en.json" → "en"
            else:
                # This is likely "fables.json" with no language code
                lang = ""
            
            # For debugging
            if not lang:
                logging.info(f"Found file with empty or missing language code: {lang_file.name}")
                
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    # Parse the JSON file
                    fables = json.load(f)
                    
                    # Debug print to see what we're getting
                    print(f"Loaded {len(fables)} entries from {lang_file.name}")
                    if fables:
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

    def _fix_language_codes(self):
        """Fix empty language codes based on content analysis."""
        if '' in self.fables_by_language:
            empty_lang_fables = self.fables_by_language['']
            print(f"Found {len(empty_lang_fables)} fables with empty language code")
            
            # Analyze the fables to determine their language
            detected_fables = {}
            for fable in empty_lang_fables:
                # Check for indicators of language
                source = fable.get('source', '').lower()
                title = fable.get('title', '').lower()
                
                # Detect English fables
                detected_lang = None
                if "laura gibbs" in source:
                    detected_lang = 'en'
                    print(f"Detected English fable: {title}")
                elif "aesop" in source and not any(word in title for word in ['λύκος', 'μῦς', 'zorro']):
                    # English is likely if source mentions Aesop and title doesn't have Greek/Spanish words
                    detected_lang = 'en'
                    print(f"Likely English fable: {title}")
                
                # Add to appropriate language group
                if detected_lang:
                    if detected_lang not in detected_fables:
                        detected_fables[detected_lang] = []
                    fable['language'] = detected_lang  # Set the language field
                    detected_fables[detected_lang].append(fable)
            
            # Add detected fables to their proper language groups
            for lang, fables in detected_fables.items():
                if lang not in self.fables_by_language:
                    self.fables_by_language[lang] = []
                self.fables_by_language[lang].extend(fables)
                print(f"Added {len(fables)} fables to language '{lang}'")
            
            # Remove the empty language key if we've fixed all fables
            fixed_count = sum(len(fables) for fables in detected_fables.values())
            if fixed_count == len(empty_lang_fables):
                del self.fables_by_language['']
                print("Removed empty language group after fixing all fables")
            else:
                # Update the empty language group
                remaining = [f for f in empty_lang_fables if not any(f in fables for fables in detected_fables.values())]
                self.fables_by_language[''] = remaining
                print(f"Still have {len(remaining)} fables with undetected language")

    def serialize_spacy_objects(self, obj, visited=None):
        """Recursively serialize spaCy objects to JSON-compatible data structures."""
        # Initialize the visited set on the first call
        if visited is None:
            visited = set()
        
        # Get the object ID to detect cycles
        obj_id = id(obj)
        
        # If we've seen this object before, return a placeholder to break the cycle
        if obj_id in visited:
            return "<circular reference>"
        
        # Add this object to the visited set
        visited.add(obj_id)
        
        # Handle different types
        if obj is None or isinstance(obj, (int, float, str, bool)):
            # Base types can be returned as is
            return obj
        
        elif isinstance(obj, list) or isinstance(obj, tuple):
            # For lists and tuples, recursively serialize each item
            result = [self.serialize_spacy_objects(item, visited.copy()) for item in obj]
            # For debugging empty POS tags
            if not result and isinstance(obj, list) and str(obj).find('pos_tags') > -1:
                print(f"Warning: Empty result after serializing what appears to be POS tags: {obj}")
            return result
        
        elif isinstance(obj, dict):
            # For dictionaries, recursively serialize each value
            return {key: self.serialize_spacy_objects(value, visited.copy()) for key, value in obj.items()}
        
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            # If the object has a to_dict method, use that
            return self.serialize_spacy_objects(obj.to_dict(), visited.copy())
        
        elif hasattr(obj, '__dict__'):
            # For objects with a __dict__, serialize the dictionary
            # but filter out private attributes (those starting with _)
            filtered_dict = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            return self.serialize_spacy_objects(filtered_dict, visited.copy())
        
        else:
            # For anything else, convert to string
            try:
                return str(obj)
            except:
                return "<unserializable object>"

    def clean_fable_text(self, fable: Dict[str, Any]) -> Dict[str, Any]:
        """Clean fable text by removing XML tags and normalizing content."""
        if 'body' in fable:
            body = fable['body']
            # Clean XML-like tags
            body = re.sub(r'</?body>', '', body)
            body = re.sub(r'<moral.*?</moral>', '', body, flags=re.DOTALL)
            # Normalize quotes
            body = body.replace('"', '"').replace('"', '"')
            # Trim whitespace
            body = body.strip()
            # Update the fable
            fable['body'] = body
        return fable

    def process_all_languages(self):
        """Process fables in all languages."""
        for lang, fables in self.fables_by_language.items():
            # Get appropriate model
            nlp = get_model(lang)
            if not nlp:
                print(f"Skipping {lang} due to missing model")
                continue
                
            # Special diagnostics for English
            if lang == 'en':
                print(f"\n🔍 ENGLISH PROCESSING DIAGNOSTICS:")
                print(f"  - Using model: {nlp.meta.get('name', 'unknown')}")
                print(f"  - Found {len(fables)} English fables")
                
                # Check first fable content 
                if fables:
                    first_fable = fables[0]
                    print(f"  - First fable: ID={first_fable.get('id')}, Title={first_fable.get('title')}")
                    body = first_fable.get('body', '')
                    print(f"  - Body length: {len(body)} chars")
                    print(f"  - Body starts with: {body[:50]}")
                    
                    # Check for XML tags
                    if re.search(r'</?body>|</?moral', body):
                        print("  - ⚠️ XML tags found in body text - will clean")
            
            print(f"Processing {len(fables)} fables in language: {lang}")
            
            # Process each fable
            processed_fables = []
            for fable in fables:
                try: 
                    # Clean the fable text first
                    cleaned_fable = self.clean_fable_text(fable)
                    # Process the fable
                    processed = preprocess_fable(cleaned_fable, nlp)
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

        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                fables = json.load(f)

            print(f"Analyzing POS distribution for {language} ({len(fables)} fables)")
            
            # Special diagnostics for English
            if language == 'en':
                for i, fable in enumerate(fables[:2]):  # Check first two
                    print(f"  Fable {i+1}: {fable.get('title', 'Unknown')}")
                    pos_tags = fable.get('pos_tags', [])
                    print(f"  - POS tags count: {len(pos_tags)}")
                    if pos_tags:
                        print(f"  - Sample tags: {pos_tags[:5]}")
                    else:
                        print(f"  - ⚠️ No POS tags found!")

            # Count POS tags
            for fable in fables:
                pos_tags = fable.get('pos_tags', [])
                for _, pos in pos_tags:
                    pos_counts[pos] += 1
                    total_tokens += 1

            print(f"Found {total_tokens} tokens with POS tags for {language}")
            
            # Handle no tokens case
            if total_tokens == 0:
                print(f"⚠️ No tokens found for {language} - check processing pipeline")
                return {}
                
            # Convert to percentages and sort by frequency
            return {pos: count/total_tokens*100 for pos, count in pos_counts.most_common()}
            
        except Exception as e:
            print(f"Error analyzing POS distribution for {language}: {e}")
            return {}

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
        
        # Check which languages are missing this fable
        all_langs = set(self.fables_by_language.keys()) - {''}  # Exclude empty language code
        missing_langs = all_langs - set(languages)
        if missing_langs:
            print(f"Note: Fable {fable_id} is missing in languages: {', '.join(missing_langs)}")
        
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
                print(f"No processed data file for language {lang}")
                continue

            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    fables = json.load(f)

                # Find the specific fable
                found = False
                for fable in fables:
                    if fable.get('id') == fable_id:
                        tokens = fable.get('tokens', [])
                        entities = fable.get('entities', [])
                        moral_text = fable.get('moral', {}).get('text', '')
                        
                        comparison['token_counts'][lang] = len(tokens)
                        comparison['entity_counts'][lang] = len(entities)
                        comparison['has_moral'][lang] = bool(moral_text)
                        
                        found = True
                        break
                
                if not found:
                    print(f"Fable {fable_id} not found in processed data for {lang}")
            
            except Exception as e:
                print(f"Error loading processed data for {lang}: {e}")
    
        return comparison