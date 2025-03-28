# src/aesop_spacy/io/loader.py
"""
Module providing functionality for loading multilingual fable data from various file formats.
Enhanced with improved language detection and error handling while maintaining compatibility.
"""
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import json
import re
import logging
from collections import defaultdict
import unicodedata


class FableLoader:
    """Responsible for loading fable data from various file formats with linguistic awareness."""

    # ISO language code validation set (common languages plus ancient ones)
    VALID_LANGS: Set[str] = {
        'en', 'de', 'nl', 'es', 'fr', 'it', 'pt', 'ru', 'zh', 'ja', 'ar', 
        'hi', 'ko', 'tr', 'pl', 'sv', 'da', 'no', 'fi', 'grc', 'la', 'sa'
    }

    # Language detection patterns - key linguistic markers 
    LANG_PATTERNS = {
        'en': (
            re.compile(r'\b(the|and|of|to|in|that|with|for)\b', re.IGNORECASE),
            ['laura gibbs', 'oxford', 'english', 'anglo']
        ),
        'de': (
            re.compile(r'\b(der|die|das|und|zu|in|den|mit|von|ein|eine)\b', re.IGNORECASE),
            ['gutenberg-de', 'deutsch', 'german']
        ),
        'nl': (
            re.compile(r'\b(de|het|een|en|van|in|op|met|te|dat|voor)\b', re.IGNORECASE),
            ['koen van den bruele', 'nederlands', 'dutch']
        ),
        'es': (
            re.compile(r'\b(el|la|los|las|un|una|y|de|en|que|por)\b', re.IGNORECASE),
            ['español', 'spanish', 'castellano']
        ),
        'grc': (
            re.compile(r'[\u0370-\u03FF]'),
            ['ancient greek', 'ελληνικά', 'greek']
        )
    }

    # Character sets that strongly indicate a specific language
    CHAR_INDICATORS = {
        'de': {'ä', 'ö', 'ü', 'ß'},
        'nl': {'ij', 'ui', 'ĳ'},
        'es': {'ñ', '¿', '¡'},
        'grc': {'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ'}
    }

    def __init__(self, data_dir: Path):
        """Initialize the loader with a root data directory."""
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
        Load all fables and organize them by language.
        
        Returns:
            Dictionary mapping language codes to lists of fable dictionaries
        """
        fables_by_language = defaultdict(list)
        
        # Check for expected language files
        expected_langs = ['en', 'de', 'nl', 'es', 'grc']
        processed_dir = self.data_dir / "processed"
        missing_langs = []
        
        # Determine which language files are missing
        if processed_dir.exists():
            for lang in expected_langs:
                json_file = processed_dir / f"fables_{lang}.json"
                if not json_file.exists():
                    missing_langs.append(lang)
                    self.logger.info("Missing language file for %s, will be processed", lang)
        else:
            # If processed directory doesn't exist, all languages need processing
            missing_langs = expected_langs.copy()
            self.logger.info("Processed directory not found, will process all languages")
        
        # First load existing JSON files
        existing_fables = self.load_json_files()
        for lang, fables in existing_fables.items():
            fables_by_language[lang].extend(fables)
        
        # Then process missing languages if needed
        if missing_langs:
            # Load markdown files first to get raw data
            markdown_file = self.data_dir / "raw" / "fables" / "initial_fables.md"
            if markdown_file.exists():
                markdown_fables = self.load_from_markdown(markdown_file)
                
                # Only process languages that are missing
                for lang in missing_langs:
                    if lang in markdown_fables:
                        fables_by_language[lang] = markdown_fables[lang]
                    else:
                        self.logger.warning("Language %s not found in markdown data", lang)
            else:
                self.logger.warning("Markdown file not found: %s", markdown_file)
            
            # Fix language codes and validate fables
            self._fix_language_codes(fables_by_language)
            
            # Create processed directory if it doesn't exist
            if missing_langs and not processed_dir.exists():
                processed_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Created processed directory: %s", processed_dir)
            
            # Save processed files for missing languages
            for lang in missing_langs:
                if lang in fables_by_language:
                    output_file = processed_dir / f"fables_{lang}.json"
                    try:
                        output_file.write_text(
                            json.dumps(fables_by_language[lang], ensure_ascii=False, indent=2),
                            encoding='utf-8'
                        )
                        self.logger.info("Created processed file for language %s: %s", lang, output_file)
                    except Exception as e:
                        self.logger.error("Error saving %s: %s", output_file, e)
        
        return dict(fables_by_language)

    def load_json_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load fables from JSON files in the processed directory."""
        fables_by_language = defaultdict(list)
        processed_dir = self.data_dir / "processed"
        
        if not processed_dir.exists():
            self.logger.warning("Processed directory not found: %s", processed_dir)
            return {}
        
        for json_file in processed_dir.glob("fables*.json"):
            lang = self._extract_language_from_filename(json_file)
            
            try:
                fables = json.loads(json_file.read_text(encoding='utf-8'))
                
                if not isinstance(fables, list):
                    self.logger.warning("Expected list but got %s in %s", type(fables), json_file)
                    continue
                    
                self.logger.info("Loaded %d fables from %s", len(fables), json_file.name)
                fables_by_language[lang].extend(fables)
                
            except Exception as e:
                self.logger.error("Error loading %s: %s: %s", json_file.name, type(e).__name__, e)
        
        return dict(fables_by_language)

    def load_from_markdown(self, markdown_file: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and parse fables from a markdown file with fable sections."""
        try:
            content = markdown_file.read_text(encoding='utf-8')
        except Exception as e:
            self.logger.error("Error reading markdown file %s: %s: %s", markdown_file, type(e).__name__, e)
            return {}
        
        fables_by_language = defaultdict(list)
        
        # Find language version sections
        language_sections = re.findall(
            r'#{3}\s+(.*?)\s+Version\s*\n(.*?)(?=#{3}|\Z)', 
            content, re.DOTALL
        )
        
        fable_count = 0
        
        for language_name, fable_content in language_sections:
            # Extract language code
            lang_match = re.search(r'<language>(.*?)</language>', fable_content)
            if not lang_match:
                self.logger.warning("No language tag found for %s version", language_name)
                continue
                
            lang = lang_match.group(1)
            
            # Extract fable data
            fable = {
                'title': self._extract_tag(fable_content, 'title', ''),
                'language': lang,
                'source': self._extract_tag(fable_content, 'source', ''),
                'version': self._extract_tag(fable_content, 'version', '1'),
                'body': self._extract_tag(fable_content, 'body', ''),
                'id': self._extract_tag(fable_content, 'fable_id', '')
            }
            
            # Handle moral
            moral_match = re.search(
                r'<moral\s+type="(.*?)">(.*?)</moral>', 
                fable_content, re.DOTALL
            )
            
            if moral_match:
                fable['moral'] = {
                    'type': moral_match.group(1),
                    'text': moral_match.group(2).strip()
                }
            else:
                moral_text = self._extract_tag(fable_content, 'moral', '')
                if moral_text:
                    fable['moral'] = {
                        'type': 'unknown',
                        'text': moral_text
                    }
            
            fables_by_language[lang].append(fable)
            fable_count += 1
        
        total_fables = sum(len(fables) for fables in fables_by_language.values())
        self.logger.info("Loaded %d fables from markdown across %d languages", 
                         total_fables, len(fables_by_language))
        
        return dict(fables_by_language)

    def _detect_language(self, fable: Dict[str, Any]) -> Optional[str]:
        """
        Detect the language of a fable using linguistic patterns and content analysis.
        Returns ISO language code or None if detection fails.
        """
        # Extract text for analysis, normalize Unicode for consistency
        source = unicodedata.normalize('NFC', fable.get('source', '').lower())
        title = unicodedata.normalize('NFC', fable.get('title', '').lower())
        body = unicodedata.normalize('NFC', fable.get('body', '').lower())
        
        # Combine text for analysis
        all_text = f"{title} {body} {source}"
        
        # Check for explicit language markers in source
        if "laura gibbs" in source or "oxford" in source:
            return 'en'
        
        if "gutenberg" in source and any(word in title for word in ['der', 'das', 'die']):
            return 'de'
        
        if "koen van den bruele" in source:
            return 'nl'
        
        # Character set analysis (fastest check)
        for lang, chars in self.CHAR_INDICATORS.items():
            for char in chars:
                if char in all_text:
                    return lang
        
        # Greek alphabet check (reliable indicator for Ancient Greek)
        if re.search(r'[\u0370-\u03FF]', all_text):
            return 'grc'
        
        # Check for language-specific patterns (more CPU intensive)
        lang_scores = {}
        
        for lang, (pattern, keywords) in self.LANG_PATTERNS.items():
            # Ensure pattern is a compiled regex
            if hasattr(pattern, 'findall'):
                # Pattern is already a compiled regex object
                pattern_matches = len(pattern.findall(all_text))
            else:
                # Pattern is a string, compile it first
                compiled_pattern = re.compile(pattern)
                pattern_matches = len(compiled_pattern.findall(all_text))
            
            # Count keyword matches
            keyword_matches = sum(keyword in all_text for keyword in keywords)
            
            # Calculate weighted score
            lang_scores[lang] = pattern_matches * 2 + keyword_matches * 5
        
        # Return language with highest score if it exceeds threshold
        best_lang = max(lang_scores.items(), key=lambda x: x[1], default=(None, 0))
        
        if best_lang[1] >= 5:  # Require minimum score threshold
            return best_lang[0]
        
        # Default to 'en' for Aesop Collection if nothing else matched
        if "aesop" in source and "collection" in source:
            return 'en'
        
        return None

    def _fix_language_codes(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> None:
        """Detect and fix empty language codes based on content analysis."""
        # Fix missing language codes
        if '' in fables_by_language:
            empty_lang_fables = fables_by_language['']
            self.logger.info("Found %d fables with empty language code", len(empty_lang_fables))
            
            # Track which fables need to be reassigned
            to_reassign = []
            
            # Detect languages for fables with empty code
            for fable in empty_lang_fables:
                detected_lang = self._detect_language(fable)
                
                if detected_lang:
                    # Update the fable and mark for reassignment
                    fable['language'] = detected_lang
                    to_reassign.append((fable, detected_lang))
                    self.logger.info("Detected language '%s' for fable: %s", 
                                    detected_lang, fable.get('title', 'Untitled'))
            
            # Move fables to their detected language groups
            for fable, lang in to_reassign:
                fables_by_language[lang].append(fable)
                empty_lang_fables.remove(fable)
            
            # Clean up empty language group if needed
            if not empty_lang_fables:
                del fables_by_language['']
                self.logger.info("All fables reassigned to language-specific groups")
            else:
                self.logger.info("%d fables remain with undetected language", 
                               len(empty_lang_fables))
        
        # Validate language codes against standards
        for lang in list(fables_by_language.keys()):
            if lang not in self.VALID_LANGS and lang != '':
                self.logger.warning("Non-standard language code: '%s'", lang)
        
        # Basic data validation for all fables
        for lang, fables in fables_by_language.items():
            for i, fable in enumerate(fables):
                if not fable.get('body'):
                    self.logger.warning("Fable %d in language '%s' has empty body", i, lang)
                
                if not fable.get('title'):
                    self.logger.warning("Fable %d in language '%s' has no title", i, lang)

    def _extract_language_from_filename(self, file_path: Path) -> str:
        """Extract language code from a filename like fables_XX.json."""
        parts = file_path.stem.split('_')
        return parts[1] if len(parts) > 1 else ""

    def _extract_tag(self, content: str, tag_name: str, default: str = "") -> str:
        """Extract content from an XML-like tag."""
        match = re.search(f'<{tag_name}>(.*?)</{tag_name}>', content, re.DOTALL)
        return match.group(1).strip() if match else default