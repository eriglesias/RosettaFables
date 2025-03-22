# src/aesop_spacy/io/loader.py
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import json
import re
import logging
from collections import defaultdict


class FableLoader:
    """Responsible for loading fable data from various file formats."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the loader.
        
        Args:
            data_dir: Root data directory
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        # Configure logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all fables from both JSON and Markdown files.
        
        Returns:
            Dictionary of fables grouped by language
        """
        # Start with empty collections
        fables_by_language = defaultdict(list)

        # Load JSON files
        json_fables = self.load_json_files()
        for lang, fables in json_fables.items():
            fables_by_language[lang].extend(fables)

        # Load markdown files
        markdown_file = self.data_dir / "raw" / "fables" / "initial_fables.md"
        if markdown_file.exists():
            markdown_fables = self.load_from_markdown(markdown_file)
            for lang, fables in markdown_fables.items():
                fables_by_language[lang].extend(fables)

        # Fix any empty language codes
        self._fix_language_codes(fables_by_language)

        # Convert from defaultdict to regular dict
        return dict(fables_by_language)

    def load_json_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all fables from JSON files in the processed directory.
        
        Returns:
            Dictionary of fables grouped by language
        """
        fables_by_language = defaultdict(list)
        processed_dir = self.data_dir / "processed"
        
        if not processed_dir.exists():
            self.logger.warning(f"Processed directory does not exist: {processed_dir}")
            return {}
        
        # Process all JSON files in the directory
        for json_file in processed_dir.glob("fables*.json"):
            lang = self._extract_language_from_filename(json_file)
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    fables = json.load(f)
                
                if not isinstance(fables, list):
                    self.logger.warning(f"Expected list but got {type(fables)} in {json_file}")
                    continue
                    
                self.logger.info(f"Loaded {len(fables)} fables from {json_file.name}")
                fables_by_language[lang].extend(fables)
                
            except Exception as e:
                self.logger.error(f"Error loading {json_file.name}: {e}")
        
        return dict(fables_by_language)
    
    def load_from_markdown(self, markdown_file: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load fables from a markdown file structured with fable tags.
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            Dictionary of fables grouped by language
        """
        self.logger.info(f"Loading fables from markdown file: {markdown_file}")
        
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read markdown file {markdown_file}: {e}")
            return {}
        
        fables_by_language = defaultdict(list)
        
        # Pattern to find each language version section
        language_sections = re.findall(r'#{3}\s+(.*?)\s+Version\s*\n(.*?)(?=#{3}|\Z)', 
                                     content, re.DOTALL)
        
        fable_count = 0
        
        for language_name, fable_content in language_sections:
            # Extract language code
            lang_match = re.search(r'<language>(.*?)</language>', fable_content)
            if not lang_match:
                self.logger.warning(f"No language tag found for {language_name} version")
                continue
                
            lang = lang_match.group(1)
            
            # Extract other metadata and content
            fable = {
                'title': self._extract_tag(fable_content, 'title', ''),
                'language': lang,
                'source': self._extract_tag(fable_content, 'source', ''),
                'version': self._extract_tag(fable_content, 'version', '1'),
                'body': self._extract_tag(fable_content, 'body', ''),
                'id': self._extract_tag(fable_content, 'fable_id', '')
            }
            
            # Handle moral (which can be either explicit or implicit)
            moral_match = re.search(r'<moral\s+type="(.*?)">(.*?)</moral>', fable_content, re.DOTALL)
            if moral_match:
                fable['moral'] = {
                    'type': moral_match.group(1),
                    'text': moral_match.group(2).strip()
                }
            else:
                # Simple moral tag without type attribute
                moral_text = self._extract_tag(fable_content, 'moral', '')
                if moral_text:
                    fable['moral'] = {
                        'type': 'unknown',
                        'text': moral_text
                    }
            
            fables_by_language[lang].append(fable)
            fable_count += 1
            
        self.logger.info(f"Loaded {fable_count} fables from markdown file across {len(fables_by_language)} languages")
        
        return dict(fables_by_language)
    
    def _extract_language_from_filename(self, file_path: Path) -> str:
        """
        Extract language code from a filename following the pattern fables_XX.json.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Language code or empty string if not found
        """
        parts = file_path.stem.split('_')
        
        if len(parts) > 1:
            return parts[1]  # Normal case: "fables_en.json" → "en"
        else:
            return ""  # Unknown language
    
    def _extract_tag(self, content: str, tag_name: str, default: str = "") -> str:
        """
        Extract content from an XML-like tag.
        
        Args:
            content: Text containing XML-like tags
            tag_name: Name of the tag to extract
            default: Default value if tag is not found
            
        Returns:
            Content of the tag or default if not found
        """
        match = re.search(f'<{tag_name}>(.*?)</{tag_name}>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return default
    
    def _detect_language(self, fable: Dict[str, Any]) -> Optional[str]:
        """
        Detect the language of a fable based on its content.
        
        Args:
            fable: Fable dictionary
            
        Returns:
            Detected language code or None if detection failed
        """
        source = fable.get('source', '').lower()
        title = fable.get('title', '').lower()
        
        # Use reliable language detection patterns
        if "laura gibbs" in source:
            return 'en'
        elif "aesop" in source and "collection" in source:
            # Check for Greek words
            if any(word in title for word in ['λύκος', 'μῦς', 'κώνωψ']):
                return 'grc'
            # Check for Spanish words
            elif any(word in title for word in ['zorro', 'lobo', 'ratón']):
                return 'es'
            # Likely English by default
            elif not any(word in title for word in ['λύκος', 'μῦς', 'zorro', 'vos']):
                return 'en'
        elif "gutenberg" in source and any(word in title for word in ['der', 'das', 'die']):
            return 'de'
        elif "koen van den bruele" in source or any(word in title for word in ['de vos', 'de wolf', 'het geitje']):
            return 'nl'
        
        # Additional language-specific patterns
        body = fable.get('body', '').lower()
        
        # German patterns
        if any(word in body[:100] for word in ['ein', 'eine', 'der', 'die', 'das', 'und', 'ist']):
            if any(char in body for char in ['ä', 'ö', 'ü', 'ß']):
                return 'de'
                
        # Dutch patterns
        if any(word in body[:100] for word in ['een', 'het', 'op', 'en', 'zij', 'zijn']):
            if 'ij' in body or 'ui' in body:
                return 'nl'
                
        # Spanish patterns
        if any(word in body[:100] for word in ['un', 'una', 'el', 'la', 'los', 'las', 'y', 'es']):
            if any(char in body for char in ['ñ', '¿', '¡']):
                return 'es'
                
        # Greek - if it contains Greek characters
        if re.search(r'[\u0370-\u03FF]', body):
            return 'grc'
            
        return None
    
    def _fix_language_codes(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Detect and fix empty language codes based on content analysis.
        
        Args:
            fables_by_language: Dictionary of fables grouped by language
        """
        if '' not in fables_by_language or not fables_by_language['']:
            return
            
        empty_lang_fables = fables_by_language['']
        self.logger.info(f"Found {len(empty_lang_fables)} fables with empty language code")
        
        # Track fables that have been reassigned
        reassigned_fables: Set[int] = set()
        
        # Analyze each fable to determine its language
        for i, fable in enumerate(empty_lang_fables):
            detected_lang = self._detect_language(fable)
            
            if detected_lang:
                # Update the fable with detected language
                fable['language'] = detected_lang
                
                # Add to the correct language group
                if detected_lang not in fables_by_language:
                    fables_by_language[detected_lang] = []
                    
                fables_by_language[detected_lang].append(fable)
                reassigned_fables.add(i)
                
                self.logger.info(f"Detected language '{detected_lang}' for fable: {fable.get('title', 'Untitled')}")
        
        # Remove reassigned fables from the empty language group
        if len(reassigned_fables) == len(empty_lang_fables):
            # All fables were reassigned, remove the empty key
            del fables_by_language['']
            self.logger.info("Removed empty language group after fixing all fables")
        else:
            # Keep only unassigned fables in the empty group
            fables_by_language[''] = [f for i, f in enumerate(empty_lang_fables) 
                                     if i not in reassigned_fables]
            self.logger.info(f"Still have {len(fables_by_language[''])} fables with undetected language")