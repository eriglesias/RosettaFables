# src/aesop_spacy/preprocessing/cleaner.py
"""
Text cleaning and normalization utilities for fable text analysis.

This module provides functionality for cleaning and normalizing fable text
across multiple languages, with special handling for capitalization inconsistencies,
character name normalization, and language-specific text preprocessing.

It includes functionality to:
- Normalize whitespace and punctuation
- Fix common encoding issues
- Handle language-specific quote formatting
- Normalize character name capitalization for consistent entity recognition
- Create canonical forms for consistent text analysis
"""

from typing import Dict, Any, Optional
import re
import logging


class TextCleaner:
    """Cleans and normalizes fable text with special handling for different languages."""

    def __init__(self):
        """Initialize the text cleaner with logger."""
        self.logger = logging.getLogger(__name__)
        self.canonical_forms = {}  # Store canonical forms for each document

    def clean_fable(self, fable: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean an entire fable dictionary by normalizing text fields.
        
        Args:
            fable: Dictionary containing fable data
            
        Returns:
            Dictionary with cleaned text fields
        """
        if not fable:
            self.logger.warning("Empty fable provided to cleaner")
            return {}

        # Create a copy to avoid modifying the original
        cleaned_fable = fable.copy()

        # Get language for language-specific cleaning
        language = fable.get('language', '').lower()

        # Clean text fields
        for field in ['body', 'title']:
            if field in fable and fable[field]:
                cleaned_fable[field] = self._clean_text_field(fable[field], language)

        # Handle moral field which could be a dict or string
        if 'moral' in fable:
            cleaned_fable['moral'] = self._clean_moral_field(fable['moral'], language)

        # Generate canonical forms for the fable body
        if 'body' in cleaned_fable and cleaned_fable['body']:
            canonical_forms = self.create_canonical_forms(cleaned_fable['body'], language)
            cleaned_fable['canonical_forms'] = canonical_forms
            self.canonical_forms = canonical_forms  # Store for potential later use

        return cleaned_fable

    def _clean_text_field(self, text: str, language: str) -> str:
        """
        Clean a text field with proper normalization and language-specific handling.
        
        Args:
            text: Text to clean
            language: Language code for language-specific cleaning
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Basic normalization first
        text = self.normalize_text(text)
        text = self.remove_xml_tags(text)
        text = self.fix_encoding_issues(text)

        # Apply language-specific cleaning
        if language == 'en':
            text = self._clean_english_text(text)
        elif language in ['nl', 'de', 'es', 'grc']:
            # For Dutch, German, Spanish, Greek - handle general quote formatting
            text = self._clean_multilingual_quotes(text)

        # Normalize character names to ensure consistent capitalization
        text = self.normalize_character_names(text, language)

        return text

    def _clean_moral_field(self, moral: Any, language: str) -> Any:
        """
        Clean the moral field which could be a dict or string.
        
        Args:
            moral: Moral field value (dict or string)
            language: Language code
            
        Returns:
            Cleaned moral field value
        """
        if isinstance(moral, dict):
            moral_copy = moral.copy()
            if 'text' in moral and moral['text']:
                moral_copy['text'] = self._clean_text_field(moral['text'], language)
            return moral_copy
        elif isinstance(moral, str):
            # If moral is a string, convert to dict with cleaned text
            return {
                'type': 'unknown',
                'text': self._clean_text_field(moral, language)
            }
        else:
            return moral

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by fixing whitespace and standardizing characters.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Remove extra whitespace (including non-breaking spaces)
        text = re.sub(r'[\s\xa0]+', ' ', text).strip()

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')

        # Normalize ellipses
        text = text.replace('…', '...')

        # Ensure space after punctuation but not before
        text = re.sub(r'([.,;:!?])(?!\s)', r'\1 ', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_xml_tags(self, text: str) -> str:
        """
        Remove XML tags from text.
        
        Args:
            text: Text containing XML tags
            
        Returns:
            Text with XML tags removed
        """
        if not text:
            return ""

        text = re.sub(r'<[^>]+>', '', text)

        return text.strip()

    def fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding issues in text.
        
        Args:
            text: Text with potential encoding issues
            
        Returns:
            Text with encoding issues fixed
        """
        if not text:
            return ""

        
        replacements = {
            '\u00A0': ' ',    # Non-breaking space
            '\u2022': '•',    # Bullet
            '\u2013': '-',    # En dash
            '\u2014': '-',    # Em dash
            '\u2018': "'",    # Left single quote
            '\u2019': "'",    # Right single quote
            '\u201C': '"',    # Left double quote
            '\u201D': '"',    # Right double quote
            '\u2026': '...',  # Ellipsis
            '\u00AB': '"',    # Left pointing double angle quotation mark
            '\u00BB': '"',    # Right pointing double angle quotation mark
        }

        for orig, repl in replacements.items():
            text = text.replace(orig, repl)

        return text

    def _clean_english_text(self, text: str) -> str:
        """
        Apply English-specific text cleaning rules, especially for quotes.
        
        Args:
            text: English text to clean
            
        Returns:
            Cleaned English text
        """
        if not text:
            return ""

        # First pass: standardize all quotes to single quotes for English
        text = re.sub(r'[""]', "'", text)

        # Apply common quote cleaning (works for both single and double quotes)
        text = self._clean_quote_formatting(text, quote_char="'")

        return text

    def _clean_multilingual_quotes(self, text: str) -> str:
        """
        Clean quotes in languages that use double quotes.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text with proper quote formatting
        """
        if not text:
            return ""

        # Apply common quote cleaning with double quotes
        text = self._clean_quote_formatting(text, quote_char='"')

        return text

    def _clean_quote_formatting(self, text: str, quote_char: str) -> str:
        """
        Clean quote formatting regardless of the quote character used.
        
        Args:
            text: Text to clean
            quote_char: The quote character used (single or double)
            
        Returns:
            Text with properly formatted quotes
        """
        if not text:
            return ""

        # Escape the quote character for regex
        q = re.escape(quote_char)

        # Fix the split quote issue (e.g., " " or ' ')
        text = re.sub(f"{q} {q}", '', text)

        # Remove space before any punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        # Fix spaces after closing quotes (the most common issue)
        text = re.sub(f"{q}(\\s+)([.,;:!?])", f"{q}\\2", text)

        # Remove space between closing quote and following punctuation
        text = re.sub(f"{q}\\s+([.,;:!?])", f"{q}\\1", text)

        # Fix spaces between quotes and text
        text = re.sub(f"\\s+{q}", f"{q}", text)  # Remove space before opening quote
        text = re.sub(f"{q}\\s+", f"{q} ", text)  # Ensure exactly one space after closing quote

        # Fix space inside quotes at the end of sentences
        text = re.sub(f"(\\w)\\s+{q}\\s*", f"\\1{q} ", text)

        # Ensure correct spacing around periods in quoted speech
        text = re.sub(f"(\\w)\\.\\s+{q}", f"\\1.{q} ", text)

        # Fix any double spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def normalize_character_names(self, text: str, language: str, character_dict: Optional[Dict[str, str]] = None) -> str:
        """
        Normalize capitalization of character names in text.
        
        Args:
            text: Text containing character names
            language: Language code for language-specific handling
            character_dict: Optional dictionary mapping lowercase character names to preferred form
            
        Returns:
            Text with normalized character names
        """
        if not text:
            return ""
        
        # Use language-specific character dictionary if none provided
        if character_dict is None:
            character_dict = self._get_language_character_dict(language)
        
        # Special handling for German where all nouns are capitalized
        if language == 'de':
            # In German, preserve capitalization but ensure consistency
            return self._normalize_german_character_names(text, character_dict)
        
        # For other languages, normalize based on the character dictionary
        for char_lower, char_preferred in character_dict.items():
            # Create patterns to match the character name with various capitalizations
            patterns = [
                rf'\b{char_lower}\b',                    # all lowercase
                rf'\b{char_lower.capitalize()}\b',       # First letter capitalized
            ]
            
            # Add patterns with articles if they exist in the text
            if language == 'nl':
                # Dutch articles
                articles = ['de', 'het', 'een', 'De', 'Het', 'Een']
                for article in articles:
                    pattern = rf'\b{article}\s+{char_lower}\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        patterns.append(pattern)
            
            elif language == 'en':
                # English articles
                articles = ['the', 'a', 'an', 'The', 'A', 'An']
                for article in articles:
                    pattern = rf'\b{article}\s+{char_lower}\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        patterns.append(pattern)
            
            # Apply all patterns
            for pattern in patterns:
                if 'de ' in pattern.lower() or 'het ' in pattern.lower() or 'een ' in pattern.lower():
                    # For Dutch patterns with articles
                    article_match = re.search(r'\\b(\w+)\\s\+', pattern)
                    if article_match:
                        article = article_match.group(1)
                        text = re.sub(pattern, f'{article} {char_preferred}', text, flags=re.IGNORECASE)
                elif 'the ' in pattern.lower() or 'a ' in pattern.lower() or 'an ' in pattern.lower():
                    # For English patterns with articles
                    article_match = re.search(r'\\b(\w+)\\s\+', pattern)
                    if article_match:
                        article = article_match.group(1)
                        text = re.sub(pattern, f'{article} {char_preferred}', text, flags=re.IGNORECASE)
                else:
                    # For patterns without articles
                    text = re.sub(pattern, char_preferred, text, flags=re.IGNORECASE)
        
        return text

    def _normalize_german_character_names(self, text: str, character_dict: Dict[str, str]) -> str:
        """
        Special handling for German text where all nouns are capitalized.
        
        Args:
            text: German text
            character_dict: Dictionary of character names
            
        Returns:
            Text with normalized character names
        """
        # For German, we need to ensure all character names are capitalized
        # But we also need to ensure consistency in how they're capitalized
        
        for char_lower, char_preferred in character_dict.items():
            # In German, the preferred form should always be capitalized
            char_preferred = char_preferred.capitalize()
            
            # Match the character name with or without articles
            patterns = [
                rf'\b{char_lower}\b',                    # all lowercase
                rf'\b{char_lower.capitalize()}\b',       # First letter capitalized
            ]
            
            # Add patterns with German articles
            articles = ['der', 'die', 'das', 'ein', 'eine', 'Der', 'Die', 'Das', 'Ein', 'Eine']
            for article in articles:
                pattern = rf'\b{article}\s+{char_lower}\b'
                if re.search(pattern, text, re.IGNORECASE):
                    patterns.append(pattern)
            
            # Apply all patterns
            for pattern in patterns:
                if any(article.lower() in pattern.lower() for article in articles):
                    # For patterns with articles
                    article_match = re.search(r'\\b(\w+)\\s\+', pattern)
                    if article_match:
                        article = article_match.group(1)
                        text = re.sub(pattern, f'{article} {char_preferred}', text, flags=re.IGNORECASE)
                else:
                    # For patterns without articles
                    text = re.sub(pattern, char_preferred, text, flags=re.IGNORECASE)
        
        return text

    def _get_language_character_dict(self, language: str) -> Dict[str, str]:
        """
        Get a dictionary of character names for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary mapping lowercase character names to preferred form
        """
        # Default character dictionaries by language
        character_dicts = {
            'en': {
                "wolf": "Wolf",
                "lamb": "Lamb",
                "fox": "Fox",
                "lion": "Lion",
                "mouse": "Mouse",
                "crane": "Crane",
                "goat": "Goat",
                "mosquito": "Mosquito",
                "frog": "Frog",
                "cat": "Cat",
                "dog": "Dog",
                "city mouse": "City Mouse",
                "country mouse": "Country Mouse",
            },
            'nl': {
                "wolf": "Wolf",
                "geitje": "Geitje", 
                "vos": "Vos",
                "leeuw": "Leeuw",
                "muis": "Muis",
                "kraanvogel": "Kraanvogel",
                "geit": "Geit",
                "mug": "Mug",
                "stadsmuis": "Stadsmuis",
                "veldmuis": "Veldmuis",
            },
            'de': {
                "wolf": "Wolf",
                "lamm": "Lamm",
                "fuchs": "Fuchs",
                "löwe": "Löwe",
                "maus": "Maus",
                "kranich": "Kranich",
                "ziege": "Ziege",
                "mücke": "Mücke",
                "stadtmaus": "Stadtmaus",
                "landmaus": "Landmaus",
            },
            'es': {
                "lobo": "Lobo",
                "cordero": "Cordero",
                "zorro": "Zorro",
                "león": "León",
                "ratón": "Ratón",
                "grulla": "Grulla",
                "cabra": "Cabra",
                "mosquito": "Mosquito",
            },
        }
        
        # Return the appropriate dictionary or an empty one if language not supported
        return character_dicts.get(language, {})

    def create_canonical_forms(self, text: str, language: str) -> Dict[str, str]:
        """
        Create canonical forms for consistent capitalization throughout a text.
        This is essential for accurate entity recognition and text analysis.
        
        Args:
            text: The text to analyze for canonical forms
            language: The language code for language-specific handling
            
        Returns:
            Dictionary mapping lowercase words to their canonical forms
        """
        if not text:
            return {}
        
        # Get the language-specific character dictionary
        character_dict = self._get_language_character_dict(language)
        
        # Initialize the canonical forms dictionary with the character dictionary
        canonical_forms = {k.lower(): v for k, v in character_dict.items()}
        
        # Add special handling for German where all nouns are capitalized
        if language == 'de':
            # For German, we need to identify all nouns (which would require POS tagging)
            # As a simplification, we'll preserve capitalization for any capitalized word
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if word[0].isupper():
                    canonical_forms[word.lower()] = word
        else:
            # For other languages, track consistent capitalization for potential character names
            # Split text and track capitalization
            words = re.findall(r'\b\w+\b', text)
            
            for word in words:
                word_lower = word.lower()
                
                # Skip very short words and common function words
                if len(word) <= 2 or word_lower in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                    continue
                
                # If this word isn't already in our canonical forms
                if word_lower not in canonical_forms:
                    # For potential character names and other proper nouns, prefer capitalized forms
                    if word[0].isupper():
                        canonical_forms[word_lower] = word
                    else:
                        # For other words, use lowercase form by default
                        canonical_forms[word_lower] = word
                # If it's already in our forms but we find a capitalized version, use that instead
                elif word[0].isupper() and not canonical_forms[word_lower][0].isupper():
                    canonical_forms[word_lower] = word
        
        return canonical_forms