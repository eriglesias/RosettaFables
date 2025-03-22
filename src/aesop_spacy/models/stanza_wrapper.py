"""
Stanza wrapper module for Ancient Greek language processing.
Provides compatibility between Stanza's output and the spaCy-based pipeline.
"""
import logging

class Token:
    """Simple class to mimic essential functionality of spaCy's Token class."""
    
    def __init__(self, text, pos=None, lemma=None, i=None):
        self.text = text
        self.pos_ = pos
        self.lemma_ = lemma
        self.i = i  # Index in the doc
        self.ent_type_ = ""  # Named entity type (empty placeholder)
        self.ent_iob_ = "O"  # IOB tag (empty placeholder)
        self.is_stop = False  # Is stopword flag
        self.is_alpha = text.isalpha() if text else False
        self.dep_ = ""  # Dependency relation
        self.head = self  # Head token (self-reference by default)
        
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f"Token({self.text}, pos={self.pos_}, lemma={self.lemma_})"


class Span:
    """Simple class to mimic essential functionality of spaCy's Span class."""
    
    def __init__(self, token_indexes, text, tokens=None):
        self.text = text
        self.start = token_indexes[0] if token_indexes else 0
        self.end = token_indexes[-1] + 1 if token_indexes else 0
        self.token_indexes = token_indexes
        self._tokens = tokens  # Optional reference to actual tokens
        
    def __repr__(self):
        return f"Span({self.text})"
    
    def __iter__(self):
        """Iterate through tokens if we have them."""
        if self._tokens:
            for idx in self.token_indexes:
                if 0 <= idx < len(self._tokens):
                    yield self._tokens[idx]


class CompatibleDoc:
    """Custom document class that mimics essential spaCy Doc functionality."""
    
    def __init__(self, text):
        self.text = text
        self.tokens = []  # Will store Token objects
        self.words = []   # Alias for tokens - for compatibility with model_manager
        self.ents = []    # Named entities (empty list as placeholder)
        self.sentence_spans = []  # Will hold sentence spans
        self.vector = []  # Empty vector for compatibility
        
        # Make sentences attribute refer to sentence_spans for compatibility
        # with model_manager.py's expectations
        self.sentences = self.sentence_spans
    
    def __len__(self):
        """Return number of tokens, similar to spaCy's behavior."""
        return len(self.tokens)
        
    def __getitem__(self, key):
        """Support indexing to access tokens."""
        if isinstance(key, slice):
            # Return a list of tokens for slices
            return self.tokens[key]
        return self.tokens[key]
    
    def __iter__(self):
        """Make the document iterable over tokens."""
        return iter(self.tokens)
    
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
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, text):
        """Process text and return a compatible document object."""
        try:
            # Process with Stanza
            stanza_doc = self.pipeline(text)
            # Convert to a format compatible with your code
            return self._convert_to_compatible_format(stanza_doc, text)
        except Exception as e:
            self.logger.error(f"Error processing text with Stanza: {e}")
            # Return a minimal document with the text
            doc = CompatibleDoc(text)
            return doc
    
    def _convert_to_compatible_format(self, stanza_doc, text):
        """Convert Stanza document to a format compatible with existing code."""
        doc = CompatibleDoc(text)
        
        # Extract sentences
        token_idx = 0
        for sent_idx, sent in enumerate(stanza_doc.sentences):
            # Get sentence text
            sent_text = sent.text if hasattr(sent, 'text') else ""
            if not sent_text:
                # Try to reconstruct from words
                if hasattr(sent, 'words'):
                    sent_text = " ".join([w.text for w in sent.words])
                elif hasattr(sent, 'tokens'):
                    sent_text = " ".join([t.text for t in sent.tokens])
            
            # Approximate start and end positions
            sent_start = text.find(sent_text) if sent_text else -1
            if sent_start < 0:
                # If exact match not found, approximate
                sent_start = sum(len(getattr(s, 'text', '')) for s in stanza_doc.sentences[:sent_idx])
            sent_end = sent_start + len(sent_text) if sent_text else 0
            
            # Track token indices for sentence spans
            token_start_idx = token_idx
            token_indices = []
            
            # Extract tokens, lemmas, and POS tags
            if hasattr(sent, 'words'):
                # Stanza typically puts individual words in a 'words' attribute
                for word in sent.words:
                    # Create a Token object for each word
                    token = Token(
                        text=getattr(word, 'text', ""),
                        pos=getattr(word, 'upos', ""),  # Universal POS tag
                        lemma=getattr(word, 'lemma', getattr(word, 'text', "")),
                        i=token_idx
                    )
                    doc.tokens.append(token)
                    doc.words.append(token)  # Add to words list for compatibility
                    token_indices.append(token_idx)
                    token_idx += 1
            elif hasattr(sent, 'tokens'):
                # Some models may have direct tokens
                for token in sent.tokens:
                    if hasattr(token, 'words'):
                        # Handle multi-word tokens
                        for word in token.words:
                            token = Token(
                                text=getattr(word, 'text', ""),
                                pos=getattr(word, 'upos', ""),
                                lemma=getattr(word, 'lemma', getattr(word, 'text', "")),
                                i=token_idx
                            )
                            doc.tokens.append(token)
                            doc.words.append(token)
                            token_indices.append(token_idx)
                            token_idx += 1
                    else:
                        # Single token
                        token = Token(
                            text=getattr(token, 'text', ""),
                            pos=getattr(token, 'pos', ""),
                            lemma=getattr(token, 'lemma', getattr(token, 'text', "")),
                            i=token_idx
                        )
                        doc.tokens.append(token)
                        doc.words.append(token)
                        token_indices.append(token_idx)
                        token_idx += 1
            
            # Create a sentence span with token indices and reference to tokens
            if token_indices:
                span = Span(token_indices, sent_text, doc.tokens)
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
    logger = logging.getLogger(__name__)
    
    # Download the model if needed (this will only happen once)
    try:
        # Try simple download first - most compatible across versions
        stanza.download('grc')
    except Exception as e:
        logger.warning(f"Note: {e}")
        logger.warning("Could not download Stanza Greek model - might be offline or already downloaded")
    
    # Initialize the Ancient Greek pipeline with all processors
    try:
        # Use simple initialization without extra parameters
        nlp = stanza.Pipeline('grc', processors='tokenize,pos,lemma')
    except Exception as e:
        logger.error(f"Error initializing Stanza pipeline: {e}")
        # Create a dummy pipeline that does minimal processing
        class DummyPipeline:
            def __call__(self, text):
                # Create a minimal Stanza-like document
                class DummyDoc:
                    def __init__(self, text):
                        self.text = text
                        
                        # Create a dummy sentence
                        class DummySentence:
                            def __init__(self, text):
                                self.text = text
                                self.words = []
                                
                                # Split by spaces for simple tokenization
                                for i, word_text in enumerate(text.split()):
                                    class DummyWord:
                                        def __init__(self, text, i):
                                            self.text = text
                                            self.id = i
                                            self.upos = "X"  # Unknown POS
                                            self.lemma = text
                                    
                                    self.words.append(DummyWord(word_text, i))
                        
                        # Split text into simple sentences by punctuation
                        import re
                        sentences = re.split(r'[.!?]+', text)
                        self.sentences = [DummySentence(s.strip()) for s in sentences if s.strip()]
                
                return DummyDoc(text)
        
        nlp = DummyPipeline()
    
    # Return the wrapper around the pipeline
    return StanzaWrapper(nlp)