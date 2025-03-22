# src/aesop_spacy/preprocessing/extractor.py
from typing import Dict, Any, Tuple
import re
import logging


class ContentExtractor:
    """Extracts relevant content from fables with language-specific handling."""
    
    def __init__(self):
        """Initialize the content extractor."""
        self.logger = logging.getLogger(__name__)
    
    def extract_content(self, fable: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and organize fable content.
        
        Args:
            fable: A fable dictionary
            
        Returns:
            Fable with extracted content
        """
        # Create a copy to avoid modifying the original
        extracted = fable.copy()
        
        # Extract the fable body
        if 'body' in fable:
            extracted['extracted_body'] = self.extract_fable_body(fable)
            
        # Extract the moral if present
        if 'moral' in fable:
            extracted['extracted_moral'] = self.extract_moral(fable)
            
        return extracted
    
    def extract_fable_body(self, fable: Dict[str, Any]) -> str:
        """
        Extract the actual content from a fable with language-specific handling.
        
        Args:
            fable: A fable dictionary
            
        Returns:
            Extracted body text
        """
        body = fable.get('body', '')
        language = fable.get('language', '')
        title = fable.get('title', '')
        
        # If body is empty, return early
        if not body:
            self.logger.warning(f"Empty body for fable: {title}")
            return ""
        
        # Apply language-specific extraction
        if language == 'en':
            return self._extract_english_content(body, title)
        elif language == 'de':
            return self._extract_german_content(body, title)
        elif language == 'nl':
            return self._extract_dutch_content(body, title)
        elif language == 'es':
            return self._extract_spanish_content(body, title)
        elif language == 'grc':
            return self._extract_greek_content(body, title)
        else:
            # Generic extraction for other languages
            return self._extract_generic_content(body, title)
    
    def _extract_english_content(self, body: str, title: str) -> str:
        """
        Extract content from English fables, handling quoted text.
        
        Args:
            body: Fable body text
            title: Fable title for logging
            
        Returns:
            Extracted content
        """
        self.logger.info(f"Extracting content from English fable: {title}")
        
        # Special case for English fables that have content in quotes
        if body.startswith('"') and '"' in body[1:]:
            # Find content between quotes
            match = re.match(r'"(.*?)(?:"|\Z)', body, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                self.logger.info(f"Extracted {len(extracted)} chars from quoted content")
                return extracted
        
        # If body contains nested body tags, extract the first part
        if '<body>' in body[1:]:
            parts = body.split('<body>', 1)
            if len(parts) > 1 and parts[0]:
                # Remove any trailing quotes
                clean_part = parts[0].strip('"')
                self.logger.info(f"Extracted {len(clean_part)} chars from split content")
                return clean_part
        
        # Default to the full body
        return body
    
    def _extract_german_content(self, body: str, title: str) -> str:
        """Extract content from German fables."""
        # German fables typically don't need special extraction
        return body
    
    def _extract_dutch_content(self, body: str, title: str) -> str:
        """Extract content from Dutch fables."""
        # Dutch fables typically don't need special extraction
        return body
    
    def _extract_spanish_content(self, body: str, title: str) -> str:
        """Extract content from Spanish fables."""
        # Spanish fables typically don't need special extraction
        return body
    
    def _extract_greek_content(self, body: str, title: str) -> str:
        """Extract content from Ancient Greek fables."""
        # For Greek fables, ensure we keep all the Greek text
        # Remove any leading/trailing non-Greek content
        greek_pattern = r'[\u0370-\u03FF\u1F00-\u1FFF]+'
        
        # Find the first Greek character
        start_match = re.search(greek_pattern, body)
        if start_match:
            # Start from the first Greek character
            return body[start_match.start():]
        
        return body
    
    def _extract_generic_content(self, body: str, title: str) -> str:
        """Generic content extraction for any language."""
        # Remove obvious XML/HTML tags
        clean_body = re.sub(r'<[^>]+>', '', body)
        
        # Remove excess whitespace
        clean_body = re.sub(r'\s+', ' ', clean_body).strip()
        
        return clean_body
    
    def extract_moral(self, fable: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract moral text and metadata.
        
        Args:
            fable: A fable dictionary
            
        Returns:
            Dictionary with moral text and metadata
        """
        moral_data = {'text': '', 'type': 'unknown'}
        
        # Handle dict format
        if isinstance(fable.get('moral'), dict):
            moral_dict = fable['moral']
            moral_data['text'] = moral_dict.get('text', '')
            moral_data['type'] = moral_dict.get('type', 'unknown')
            
        # Handle string format
        elif isinstance(fable.get('moral'), str):
            moral_data['text'] = fable['moral']
            
            # Try to determine if it's explicit or implicit
            if moral_data['text'].startswith(('The moral', 'This fable')):
                moral_data['type'] = 'explicit'
            
        # Look for moral in the body text as a fallback
        elif 'body' in fable:
            body = fable['body']
            
            # Look for explicit moral indicators
            moral_patterns = [
                r'The moral of this story is[:\s]+([^\.]+\.)',
                r'Moral[:\s]+([^\.]+\.)',
                r'This fable teaches us that ([^\.]+\.)'
            ]
            
            for pattern in moral_patterns:
                match = re.search(pattern, body, re.IGNORECASE)
                if match:
                    moral_data['text'] = match.group(1).strip()
                    moral_data['type'] = 'extracted'
                    break
        
        return moral_data