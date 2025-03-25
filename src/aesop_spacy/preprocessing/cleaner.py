# src/aesop_spacy/preprocessing/cleaner.py
from typing import Dict, Any
import re
import logging


class TextCleaner:
    """Cleans and normalizes fable text with special handling for different languages."""

    def __init__(self):
        """Initialize the text cleaner with logger."""
        self.logger = logging.getLogger(__name__)

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