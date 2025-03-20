from typing import Dict, Any
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Handle empty text
    if not text:
        return ""
        
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('–', '-').replace('—', '-')
    return text

def extract_fable_content(fable: Dict[str, Any]) -> str:
    """
    Extract the actual content from fables, with special handling for English fables.
    
    Args:
        fable: Dictionary containing fable data
        
    Returns:
        Cleaned fable body text
    """
    body = fable.get('body', '')
    language = fable.get('language', '')
    title = fable.get('title', '')
    
    # Special handling for English fables
    if language == 'en':
        logging.info(f"Processing English fable: {title}")
        logging.info(f"Original body length: {len(body)}")
        
        # Extract content between quotes if present (common in English fables)
        if body.startswith('"') and '"' in body[1:]:
            # Find the content between the opening quote and the second <body> tag
            match = re.match(r'"(.*?)(?:<body>|$)', body, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logging.info(f"Extracted {len(extracted)} chars from quoted content")
                return extracted
        
        # If the body contains nested body tags, extract the first part
        if '<body>' in body[1:]:
            parts = body.split('<body>', 1)
            if len(parts) > 1 and parts[0]:
                logging.info(f"Extracted {len(parts[0])} chars from split content")
                # Remove any trailing quotes that might be part of the format
                return parts[0].strip('"')
    
    # For other languages or if special extraction failed
    return body

def preprocess_fable(fable: Dict[str, Any], nlp) -> Dict[str, Any]:
    """
    Process a fable with spaCy and extract linguistic features.
    
    Args:
        fable: Dictionary containing fable data
        nlp: Loaded spaCy model
        
    Returns:
        Fable with added linguistic features
    """
    # Create a copy to avoid modifying the original
    processed = fable.copy()
    
    # Extract and clean the fable content
    body = extract_fable_content(fable)
    
    # Debug for English fables
    if fable.get('language') == 'en':
        logging.info(f"Processing fable '{fable.get('title')}' with extracted length: {len(body)}")
        if body:
            logging.info(f"Content starts with: {body[:50]}...")
        else:
            logging.warning(f"Empty body for fable '{fable.get('title')}'")
    
    # Clean the text
    body = clean_text(body)
    
    # Only process if there's actual content
    if body:
        doc = nlp(body)
        
        # Add basic linguistic features
        processed['doc_length'] = len(doc)
        processed['tokens'] = [(token.text, token.i) for token in doc]  # Include token index
        processed['lemmas'] = [(token.lemma_, token.i) for token in doc]
        processed['pos_tags'] = [(token.text, token.pos_) for token in doc]
        processed['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Debug print token count
        if fable.get('language') == 'en':
            logging.info(f"Found {len(processed['tokens'])} tokens in '{fable.get('title')}'")
    else:
        # Set empty values if no content
        processed['doc_length'] = 0
        processed['tokens'] = []
        processed['lemmas'] = []
        processed['pos_tags'] = []
        processed['entities'] = []
        logging.warning(f"No content to process for fable '{fable.get('title')}'")

    # Process moral if present
    if 'moral' in fable and isinstance(fable['moral'], dict) and fable['moral'].get('text'):
        moral_text = clean_text(fable['moral']['text'])
        if moral_text:
            moral_doc = nlp(moral_text)
            if 'moral' not in processed:
                processed['moral'] = {}
            processed['moral']['tokens'] = [token.text for token in moral_doc]
            processed['moral']['lemmas'] = [token.lemma_ for token in moral_doc]

    return processed