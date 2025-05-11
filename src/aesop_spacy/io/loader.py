"""
Module providing functionality for loading multilingual fable data from various file formats.
Enhanced with improved language detection and error handling while maintaining compatibility.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import json
import re
import logging
from collections import defaultdict
import unicodedata


class LanguageDetector:
    """Responsible for detecting and validating language codes in fable data."""

    # ISO language code validation set (common languages plus ancient ones)
    VALID_LANGS: Set[str] = {
        'en', 'de', 'nl', 'es', 'fr', 'it', 'pt', 'ru', 'zh', 'ja', 'ar', 
        'hi', 'ko', 'tr', 'pl', 'sv', 'da', 'no', 'fi', 'grc', 'la', 'sa'
    }

    # Language detection patterns with compiled regex
    LANG_PATTERNS: Dict[str, Dict[str, Any]] = {
        'en': {
            'pattern': re.compile(r'\b(the|and|of|to|in|that|with|for)\b', re.IGNORECASE),
            'keywords': ['laura gibbs', 'oxford', 'english', 'anglo'],
            'chars': set()  # No distinctive characters for English
        },
        'de': {
            'pattern': re.compile(r'\b(der|die|das|und|zu|in|den|mit|von|ein|eine)\b', re.IGNORECASE),
            'keywords': ['gutenberg-de', 'deutsch', 'german'],
            'chars': {'ä', 'ö', 'ü', 'ß'}
        },
        'nl': {
            'pattern': re.compile(r'\b(de|het|een|en|van|in|op|met|te|dat|voor)\b', re.IGNORECASE),
            'keywords': ['koen van den bruele', 'nederlands', 'dutch'],
            'chars': {'ij', 'ui', 'ĳ'}
        },
        'es': {
            'pattern': re.compile(r'\b(el|la|los|las|un|una|y|de|en|que|por)\b', re.IGNORECASE),
            'keywords': ['español', 'spanish', 'castellano'],
            'chars': {'ñ', '¿', '¡'}
        },
        'grc': {
            'pattern': re.compile(r'[\u0370-\u03FF]'),
            'keywords': ['ancient greek', 'ελληνικά', 'greek'],
            'chars': {'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ'}
        }
    }

    def __init__(self, logger=None):
        """Initialize the language detector with an optional logger."""
        self.logger = logger or logging.getLogger(__name__)

    def detect_language(self, fable: Dict[str, Any]) -> Optional[str]:
        """
        Detect the language of a fable using linguistic patterns and content analysis.
        
        Args:
            fable: Dictionary containing fable data
            
        Returns:
            ISO language code or None if detection fails
        """
        # Extract text for analysis, normalize Unicode for consistency
        source = unicodedata.normalize('NFC', fable.get('source', '').lower())
        title = unicodedata.normalize('NFC', fable.get('title', '').lower())
        body = unicodedata.normalize('NFC', fable.get('body', '').lower())
        
        # Combine text for analysis
        all_text = f"{title} {body} {source}"
        
        # Quick source-based checks (fastest method)
        if "laura gibbs" in source or "oxford" in source:
            return 'en'
   
        if "gutenberg" in source and any(word in title for word in ['der', 'das', 'die']):
            return 'de'
  
        if "koen van den bruele" in source:
            return 'nl'
  
        # Character set analysis (fast check)
        for lang, lang_data in self.LANG_PATTERNS.items():
            chars = lang_data.get('chars', set())
            if chars and any(char in all_text for char in chars):
                return lang
    
        # Check for language-specific patterns (more CPU intensive)
        lang_scores = {}
        for lang, lang_data in self.LANG_PATTERNS.items():
            pattern = lang_data['pattern']
            keywords = lang_data['keywords']
            
            # Count pattern matches
            pattern_matches = len(pattern.findall(all_text))
            
            # Count keyword matches
            keyword_matches = sum(keyword in all_text for keyword in keywords)
            
            # Calculate weighted score
            lang_scores[lang] = pattern_matches * 2 + keyword_matches * 5
 
        # Return language with highest score if it exceeds threshold
        best_lang = max(lang_scores.items(), key=lambda x: x[1], default=(None, 0))
        if best_lang[1] >= 5:  # Minimum score threshold
            return best_lang[0]
     
        # Default to 'en' for Aesop Collection if nothing else matched
        if "aesop" in source and "collection" in source:
            return 'en'
     
        return None

    def fix_language_codes(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Detect and fix empty language codes based on content analysis.
        Validates language codes and fable data integrity.
        
        Args:
            fables_by_language: Dictionary of fables organized by language code
        """
        # Fix missing language codes
        if '' in fables_by_language:
            empty_lang_fables = fables_by_language['']
            self.logger.info("Found %d fables with empty language code", len(empty_lang_fables))
            
            # Track which fables need to be reassigned
            to_reassign = []
            
            # Detect languages for fables with empty code
            for fable in empty_lang_fables:
                detected_lang = self.detect_language(fable)
                
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


class FableLoader:
    """Responsible for loading fable data from various file formats with linguistic awareness."""

    def __init__(self, data_dir: Path):
        """
        Initialize the loader with a root data directory.
        
        Args:
            data_dir: Path to the directory containing fable data
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Create language detector
        self.language_detector = LanguageDetector(self.logger)

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all fables and organize them by language, using pre-processed JSON files
        when available and falling back to parsing raw markdown files when needed.
        
        Returns:
            Dictionary mapping language codes to lists of fable dictionaries
        """
        # Initialize storage for fables
        fables_by_language = defaultdict(list)

        # Get list of expected languages and determine which ones need processing
        expected_langs, missing_langs = self._determine_missing_languages()

        # First, load existing processed JSON files
        existing_fables = self.load_json_files()
        for lang, fables in existing_fables.items():
            fables_by_language[lang].extend(fables)

        # Process missing languages if needed
        if missing_langs:
            self._process_missing_languages(fables_by_language, missing_langs)
  
        return dict(fables_by_language)

    def _determine_missing_languages(self) -> Tuple[List[str], List[str]]:
        """
        Determine which language files need to be processed.
        
        Returns:
            Tuple of (expected languages, missing languages)
        """
        expected_langs = ['en', 'de', 'nl', 'es', 'grc']
        processed_dir = self.data_dir / "processed"
        missing_langs = []

        # Check for existing processed files
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
            
        return expected_langs, missing_langs

    def _process_missing_languages(self, 
                                  fables_by_language: Dict[str, List[Dict[str, Any]]], 
                                  missing_langs: List[str]) -> None:
        """
        Process missing language files by parsing markdown and saving as JSON.
        
        Args:
            fables_by_language: Dictionary to store loaded fables
            missing_langs: List of language codes that need processing
        """

        # Load and parse markdown file
        markdown_file = self.data_dir / "fables" / "initial_fables.md"
        if not markdown_file.exists():
            self.logger.warning("Markdown file not found: %s", markdown_file)
            return
            
        # Parse markdown and extract fables for missing languages
        markdown_fables = self.load_from_markdown(markdown_file)
        for lang in missing_langs:
            if lang in markdown_fables:
                fables_by_language[lang] = markdown_fables[lang]
            else:
                self.logger.warning("Language %s not found in markdown data", lang)
                
        # Fix language codes and validate fables
        self.language_detector.fix_language_codes(fables_by_language)
        
        # Save processed files for missing languages
        self._save_processed_languages(fables_by_language, missing_langs)

    def _save_processed_languages(self, 
                                 fables_by_language: Dict[str, List[Dict[str, Any]]], 
                                 languages: List[str]) -> None:
        """
        Save processed fables as JSON files.
        
        Args:
            fables_by_language: Dictionary of fables organized by language
            languages: List of language codes to save
        """
        processed_dir = self.data_dir / "processed"
        
        # Create processed directory if it doesn't exist
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Ensuring processed directory exists: %s", processed_dir)
        
        # Save each language to its own JSON file
        for lang in languages:
            if lang in fables_by_language:
                self._save_json_file(
                    data=fables_by_language[lang],
                    file_path=processed_dir / f"fables_{lang}.json",
                    description=f"language {lang}"
                )

    def _save_json_file(self, data: Any, file_path: Path, description: str) -> None:
        """
        Helper method to save data as JSON.
        
        Args:
            data: Data to save
            file_path: Path where file should be saved
            description: Description for logging
        """
        try:
            file_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            self.logger.info("Created processed file for %s: %s", description, file_path)
        except Exception as e:
            self.logger.error("Error saving %s: %s", file_path, e)

    def load_json_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load fables from JSON files in the processed directory.
        
        Returns:
            Dictionary mapping language codes to lists of fable dictionaries
        """
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
        """
        Extract and parse fables from a markdown file with fable sections.
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            Dictionary mapping language codes to lists of fable dictionaries
        """
        try:
            content = markdown_file.read_text(encoding='utf-8')
        except Exception as e:
            self.logger.error("Error reading markdown file %s: %s: %s", 
                             markdown_file, type(e).__name__, e)
            return {}
      
        fables_by_language = defaultdict(list)
      
        # Find language version sections
        language_sections = re.findall(
            r'#{3}\s+(.*?)\s+Version\s*\n(.*?)(?=#{3}|\Z)', 
            content, re.DOTALL
        )
        
        for language_name, fable_content in language_sections:
            fable = self._parse_fable_section(language_name, fable_content)
            if fable and 'language' in fable:
                fables_by_language[fable['language']].append(fable)

        # Log summary statistics
        total_fables = sum(len(fables) for fables in fables_by_language.values())
        self.logger.info("Loaded %d fables from markdown across %d languages", 
                        total_fables, len(fables_by_language))

        return dict(fables_by_language)

    def _parse_fable_section(self, language_name: str, fable_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single fable section from the markdown content.
        
        Args:
            language_name: Name of the language from the section header
            fable_content: Content of the fable section
            
        Returns:
            Dictionary containing parsed fable data or None if parsing fails
        """
        # Extract language code
        lang_match = re.search(r'<language>(.*?)</language>', fable_content)
        if not lang_match:
            self.logger.warning("No language tag found for %s version", language_name)
            return None
             
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
         
        # Handle moral tag which may have a type attribute
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

        return fable

    def _extract_language_from_filename(self, file_path: Path) -> str:
        """
        Extract language code from a filename like fables_XX.json.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Language code extracted from the filename
        """
        parts = file_path.stem.split('_')
        return parts[1] if len(parts) > 1 else ""

    def _extract_tag(self, content: str, tag_name: str, default: str = "") -> str:
        """
        Extract content from an XML-like tag.
        
        Args:
            content: String containing XML-like tags
            tag_name: Name of the tag to extract
            default: Default value if tag is not found
            
        Returns:
            Content of the tag or default value
        """
        match = re.search(f'<{tag_name}>(.*?)</{tag_name}>', content, re.DOTALL)
        return match.group(1).strip() if match else default