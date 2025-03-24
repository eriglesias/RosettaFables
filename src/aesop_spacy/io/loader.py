# src/aesop_spacy/io/loader.py
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import re
import logging
from collections import defaultdict


class FableLoader:
    """Responsible for loading fable data from various file formats."""

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
        """Load all fables from both JSON and Markdown files."""
        # Start with empty collections
        fables_by_language = defaultdict(list)

        # Load JSON files
        json_fables = self.load_json_files()
        for lang, fables in json_fables.items():
            fables_by_language[lang].extend(fables)

        # Load markdown files - specifically looking for initial_fables.md
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
        """Load all fables from JSON files in the processed directory."""
        fables_by_language = defaultdict(list)
        processed_dir = self.data_dir / "processed"

        if not processed_dir.exists():
            self.logger.warning("Processed directory does not exist: %s", processed_dir)
            return {}

        # Process all JSON files in the directory
        for json_file in processed_dir.glob("fables*.json"):
            lang = self._extract_language_from_filename(json_file)

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    fables = json.load(f)

                if not isinstance(fables, list):
                    self.logger.warning("Expected list but got %s in %s", 
                                      type(fables), json_file)
                    continue

                self.logger.info("Loaded %d fables from %s", len(fables), json_file.name)
                fables_by_language[lang].extend(fables)

            except FileNotFoundError:
                self.logger.error("File not found: %s", json_file)
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON in %s: %s", json_file.name, e)
            except PermissionError:
                self.logger.error("Permission denied when reading %s", json_file)
            except UnicodeDecodeError as e:
                self.logger.error("Encoding error in %s: %s", json_file.name, e)
            except IOError as e:
                self.logger.error("I/O error reading %s: %s", json_file.name, e)

        return dict(fables_by_language)

    def load_from_markdown(self, markdown_file: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Load fables from a markdown file structured with fable tags."""
        self.logger.info("Loading fables from markdown file: %s", markdown_file)

        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            self.logger.error("Markdown file not found: %s", markdown_file)
            return {}
        except PermissionError:
            self.logger.error("Permission denied when reading markdown file: %s", markdown_file)
            return {}
        except UnicodeDecodeError as e:
            self.logger.error("Encoding error in markdown file %s: %s", markdown_file, e)
            return {}
        except IOError as e:
            self.logger.error("I/O error reading markdown file %s: %s", markdown_file, e)
            return {}

        fables_by_language = defaultdict(list)

        # Pattern to find each language version section - simplified regex
        language_sections = re.findall(r'#{3}\s+(.*?)\s+Version\s*\n(.*?)(?=#{3}|\Z)', 
                                     content, re.DOTALL)

        fable_count = 0

        for language_name, fable_content in language_sections:
            # Extract language code
            lang_match = re.search(r'<language>(.*?)</language>', fable_content)
            if not lang_match:
                self.logger.warning("No language tag found for %s version", language_name)
                continue

            lang = lang_match.group(1)

            # Extract fable content with a single function call
            fable = {
                'title': self._extract_tag(fable_content, 'title', ''),
                'language': lang,
                'source': self._extract_tag(fable_content, 'source', ''),
                'version': self._extract_tag(fable_content, 'version', '1'),
                'body': self._extract_tag(fable_content, 'body', ''),
                'id': self._extract_tag(fable_content, 'fable_id', '')
            }

            # Handle moral (which can be either explicit or implicit)
            moral_match = re.search(r'<moral\s+type="(.*?)">(.*?)</moral>', 
                                  fable_content, re.DOTALL)

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

        self.logger.info("Loaded %d fables from markdown file across %d languages", 
                       fable_count, len(fables_by_language))

        return dict(fables_by_language)

    def _extract_language_from_filename(self, file_path: Path) -> str:
        """Extract language code from a filename like fables_XX.json."""
        parts = file_path.stem.split('_')
        return parts[1] if len(parts) > 1 else ""

    def _extract_tag(self, content: str, tag_name: str, default: str = "") -> str:
        """Extract content from an XML-like tag."""
        match = re.search(f'<{tag_name}>(.*?)</{tag_name}>', content, re.DOTALL)
        return match.group(1).strip() if match else default

    def _detect_language(self, fable: Dict[str, Any]) -> Optional[str]:
        """Detect the language of a fable based on its content."""
        source = fable.get('source', '').lower()
        title = fable.get('title', '').lower()
        body = fable.get('body', '').lower()

        # Fast pattern matching for common language indicators
        if "laura gibbs" in source:
            return 'en'

        if "gutenberg" in source and any(word in title for word in ['der', 'das', 'die']):
            return 'de'

        if "koen van den bruele" in source:
            return 'nl'

        # Check title for language hints
        if any(word in title for word in ['λύκος', 'μῦς', 'κώνωψ']):
            return 'grc'

        if any(word in title for word in ['zorro', 'lobo', 'ratón']):
            return 'es'

        # Body content analysis
        if re.search(r'[\u0370-\u03FF]', body):  # Greek characters
            return 'grc'

#-------------------------------------------------------
        # Language-specific patterns in content
        if any(char in body for char in ['ä', 'ö', 'ü', 'ß']):
            return 'de'

        if 'ij' in body or 'ui' in body:
            return 'nl'

        if any(char in body for char in ['ñ', '¿', '¡']):
            return 'es'

        # Default to English for Aesop Collection if no other match
        if "aesop" in source and "collection" in source:
            return 'en'

        return None

    def _fix_language_codes(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> None:
        """Detect and fix empty language codes based on content analysis."""
        if '' not in fables_by_language:
            return

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