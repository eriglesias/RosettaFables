from typing import Dict, Any
import re

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('–', '-').replace('—', '-')
    return text

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

    # Clean and process the text
    body = clean_text(fable['body'])
    doc = nlp(body)

    # Add basic linguistic features
    processed['doc_length'] = len(doc)
    processed['sentences'] = list(doc.sents)
    processed['tokens'] = [token.text for token in doc]
    processed['lemmas'] = [token.lemma_ for token in doc]
    processed['pos_tags'] = [(token.text, token.pos_) for token in doc]
    processed['entities'] = [(ent.text, ent.label_) for ent in doc.ents]

    # Process moral if present
    if 'moral' in fable and fable['moral'].get('text'):
        moral_text = clean_text(fable['moral']['text'])
        moral_doc = nlp(moral_text)
        processed['moral']['tokens'] = [token.text for token in moral_doc]
        processed['moral']['lemmas'] = [token.lemma_ for token in moral_doc]

    return processed