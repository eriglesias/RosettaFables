"""
Stanza wrapper module for Ancient Greek language processing.
Provides compatibility between Stanza's output and the spaCy-based pipeline.
"""

class Token:
    """Simple class to mimic essential functionality of spaCy's Token class."""
    
    def __init__(self, text, pos=None, lemma=None, i=None):
        self.text = text
        self.pos_ = pos
        self.lemma_ = lemma
        self.i = i  # Index in the doc
        self.ent_type_ = ""  # Named entity type (empty placeholder)
        self.ent_iob_ = ""  # IOB tag (empty placeholder)
        self.is_stop = False  # Is stopword flag
        
        # Important: we're not storing references to other objects
        # that would create cycles
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.text


class Span:
    """Simple class to mimic essential functionality of spaCy's Span class."""
    
    def __init__(self, token_indexes, text):
        self.text = text
        self.start = token_indexes[0] if token_indexes else 0
        self.end = token_indexes[-1] + 1 if token_indexes else 0
        # Store token indexes rather than tokens themselves
        self.token_indexes = token_indexes
    
    def __repr__(self):
        return self.text


class CompatibleDoc:
    """Custom document class that mimics essential spaCy Doc functionality."""
    
    def __init__(self, text):
        self.text = text
        self.tokens = []  # Will store Token objects
        self.ents = []    # Named entities (empty list as placeholder)
        self.sentence_spans = []  # Will hold sentence spans
    
    def __len__(self):
        """Return number of tokens, similar to spaCy's behavior."""
        return len(self.tokens)
        
    def __getitem__(self, key):
        """Support indexing to access tokens."""
        if isinstance(key, slice):
            # Return a list of tokens for slices
            return self.tokens[key]
        return self.tokens[key]
    
    @property
    def sents(self):
        """Return an iterator over sentence spans, similar to spaCy."""
        class SentenceIterator:
            def __init__(self, doc):
                self.doc = doc
                self.spans = doc.sentence_spans
                self.current = 0
                
            def __iter__(self):
                return self
                
            def __next__(self):
                if self.current >= len(self.spans):
                    raise StopIteration
                span = self.spans[self.current]
                self.current += 1
                return span
        
        return SentenceIterator(self)


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
        token_idx = 0
        for sent_idx, sent in enumerate(stanza_doc.sentences):
            # Get sentence text
            sent_text = sent.text
            # Approximate start and end positions
            sent_start = text.find(sent_text)
            if sent_start < 0:
                # If exact match not found, approximate
                sent_start = sum(len(s.text) for s in stanza_doc.sentences[:sent_idx])
            sent_end = sent_start + len(sent_text)
            
            # Track token indices for sentence spans
            token_start_idx = token_idx
            token_indices = []
            
            # Extract tokens, lemmas, and POS tags
            for word in sent.words:
                # Create a Token object for each word (without doc reference)
                token = Token(
                    text=word.text,
                    pos=word.upos,  # Universal POS tag
                    lemma=word.lemma if word.lemma else word.text,
                    i=token_idx
                )
                doc.tokens.append(token)
                token_indices.append(token_idx)
                token_idx += 1
            
            # Create a sentence span with just token indices
            span = Span(token_indices, sent_text)
            doc.sentence_spans.append(span)
        
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