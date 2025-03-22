# src/aesop_spacy/preprocessing/cleaner.py
from typing import Dict, Any
import re
import logging


class TextCleaner:
    """Responsible for cleaning and normalizing fable text."""
    
    def __init__(self):
        """Initialize the text cleaner."""
        self.logger = logging.getLogger(__name__)
    
    def clean_fable(self, fable: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean an entire fable by normalizing text and removing XML tags.
        
        Args:
            fable: The fable dictionary to clean
            
        Returns:
            Cleaned fable dictionary
        """
        # Create a copy to avoid modifying the original
        cleaned_fable = fable.copy()
        
        # Clean body text if present
        if 'body' in fable:
            body = fable['body']
            cleaned_body = self.normalize_text(body)
            cleaned_body = self.remove_xml_tags(cleaned_body)
            cleaned_fable['body'] = cleaned_body
            
        # Clean title if present
        if 'title' in fable:
            title = fable['title']
            cleaned_title = self.normalize_text(title)
            cleaned_fable['title'] = cleaned_title
            
        # Clean moral if present
        if 'moral' in fable:
            if isinstance(fable['moral'], dict):
                if 'text' in fable['moral']:
                    moral_text = fable['moral']['text']
                    cleaned_moral = self.normalize_text(moral_text)
                    cleaned_fable['moral']['text'] = cleaned_moral
            elif isinstance(fable['moral'], str):
                # Handle case where moral is just a string
                moral_text = fable['moral']
                cleaned_moral = self.normalize_text(moral_text)
                cleaned_fable['moral'] = {
                    'type': 'unknown',
                    'text': cleaned_moral
                }
            
        return cleaned_fable
    
    def normalize_text(self, text: str) -> str:
        """
        Clean and normalize text by fixing whitespace and special characters.
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Normalize ellipses
        text = text.replace('…', '...')
        
        # Normalize spaces after punctuation
        text = re.sub(r'([.,;:!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_xml_tags(self, text: str) -> str:
        """
        Remove XML-like tags from text.
        
        Args:
            text: Text containing XML tags
            
        Returns:
            Text with XML tags removed
        """
        if not text:
            return ""
            
        # Remove common XML tags found in fables
        text = re.sub(r'</?body>', '', text)
        text = re.sub(r'</?fable_id>.*?</fable_id>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?title>.*?</title>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?language>.*?</language>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?source>.*?</source>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?version>.*?</version>', '', text, flags=re.DOTALL)
        text = re.sub(r'<moral.*?</moral>', '', text, flags=re.DOTALL)
        
        # Remove any remaining XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        return text.strip()
    
    def fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding issues in text.
        
        Args:
            text: Text that may have encoding issues
            
        Returns:
            Text with encoding issues fixed
        """
        if not text:
            return ""
            
        # Fix common encoding issues
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