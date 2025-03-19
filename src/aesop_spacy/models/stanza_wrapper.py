"""
Stanza wrapper module for Ancient Greek language processing.
Provides compatibility between Stanza's output and the spaCy-based pipeline.
"""

class Span:
    """Simple class to mimic essential functionality of spaCy's Span class."""
    
    def __init__(self, doc, start_idx, end_idx, text):
        self.doc = doc
        self.start = start_idx
        self.end = end_idx
        self.text = text
    
    def __repr__(self):
        return self.text


class CompatibleDoc:
    """Custom document class that mimics essential spaCy Doc functionality."""
    
    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.entities = []
        self.sentences = []
        # Will hold actual Span objects for sentences
        self._sentence_spans = []
        
    def __len__(self):
        """Return number of tokens, similar to spaCy's behavior."""
        return len(self.tokens)
        
    def __getitem__(self, key):
        """Support indexing to access tokens."""
        return self.tokens[key]
    
    @property
    def sents(self):
        """Return an iterator over sentence spans, similar to spaCy."""
        return iter(self._sentence_spans)


class StanzaWrapper:
    """Wrapper for Stanza pipeline to make it compatible with spaCy interface."""
    
    def __init__(self, stanza_pipeline):
        self.pipeline = stanza_pipeline
        
    def __call__(self, text):
        """Process text and return a compatible document object."""
        # Process with Stanza
        stanza_doc = self.pipeline(text)
        # Convert to a format compatible with your code
        return self._convert_to_compatible_format(stanza_doc, text)
    
    def _convert_to_compatible_format(self, stanza_doc, text):
        """Convert Stanza document to a format compatible with existing code."""
        doc = CompatibleDoc(text)
        
        # Extract sentences
        for sent_idx, sent in enumerate(stanza_doc.sentences):
            # Get sentence text
            sent_text = sent.text
            # Approximate start and end positions
            sent_start = text.find(sent_text)
            if sent_start < 0:
                # If exact match not found, approximate
                sent_start = sum(len(s.text) for s in stanza_doc.sentences[:sent_idx])
            sent_end = sent_start + len(sent_text)
            
            # Add sentence boundaries
            doc.sentences.append({
                'text': sent_text,
                'start': sent_start,
                'end': sent_end,
                'label_': ''
            })
            
            # Track token indices for sentence spans
            token_start_idx = len(doc.tokens)
            
            # Extract tokens, lemmas, and POS tags
            for word in sent.words:
                doc.tokens.append(word.text)
                doc.lemmas.append(word.lemma if word.lemma else word.text)
                doc.pos_tags.append([word.text, word.upos])  # Using Universal POS tags
            
            # Create a sentence span from token indices
            token_end_idx = len(doc.tokens)
            span = Span(doc, token_start_idx, token_end_idx, sent_text)
            doc._sentence_spans.append(span)
        
        return doc


def get_stanza_greek_processor():
    """
    Create a Stanza processor for Ancient Greek that integrates with the existing pipeline.
    
    Returns:
        StanzaWrapper: A callable wrapper around the Stanza pipeline that returns compatible 
                      document objects.
    """
    import stanza
    from stanza.pipeline.core import DownloadMethod
    
    # Download the model if needed (this will only happen once)
    try:
        # Removed download_method parameter as it's not supported
        stanza.download('grc')
    except Exception as e:
        print(f"Note: {e}")
        print("Could not download Stanza Greek model - might be offline or already downloaded")
    
    # Initialize the Ancient Greek pipeline with all processors
    # Keep download_method here as it seems to be working
    nlp = stanza.Pipeline('grc', processors='tokenize,pos,lemma', download_method=DownloadMethod.REUSE_RESOURCES)
    
    # Return the wrapper around the pipeline
    return StanzaWrapper(nlp)