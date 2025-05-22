"""
Stanza wrapper module for Ancient Greek language processing.
Provides compatibility between Stanza's output and the spaCy-based pipeline.
"""
import logging
from typing import List, Optional, Union, Iterator, Any


class Token:
    """Simple class to mimic essential functionality of spaCy's Token class."""

    def __init__(self, text: str, pos: Optional[str] = None, 
                lemma: Optional[str] = None, i: Optional[int] = None):
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

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Token({self.text}, pos={self.pos_}, lemma={self.lemma_})"


class Span:
    """Simple class to mimic essential functionality of spaCy's Span class."""

    def __init__(self, token_indexes: List[int], text: str, tokens: Optional[List[Token]] = None):
        self.text = text
        self.start = token_indexes[0] if token_indexes else 0
        self.end = token_indexes[-1] + 1 if token_indexes else 0
        self.token_indexes = token_indexes
        self._tokens = tokens  # Optional reference to actual tokens

    def __repr__(self) -> str:
        return f"Span({self.text})"

    def __iter__(self) -> Iterator[Token]:
        """Iterate through tokens if we have them."""
        if self._tokens:
            for idx in self.token_indexes:
                if 0 <= idx < len(self._tokens):
                    yield self._tokens[idx]


class CompatibleDoc:
    """Custom document class that mimics essential spaCy Doc functionality."""

    def __init__(self, text: str):
        self.text = text
        self.tokens: List[Token] = []  # Will store Token objects
        self.words: List[Token] = []   # Alias for tokens - for compatibility with model_manager
        self.ents: List[Any] = []    # Named entities (empty list as placeholder)
        self.sentence_spans: List[Span] = []  # Will hold sentence spans
        self.vector: List[float] = []  # Empty vector for compatibility

        # Make sentences attribute refer to sentence_spans for compatibility
        # with model_manager.py's expectations
        self.sentences = self.sentence_spans

    def __len__(self) -> int:
        """Return number of tokens, similar to spaCy's behavior."""
        return len(self.tokens)

    def __getitem__(self, key: Union[int, slice]) -> Union[Token, List[Token]]:
        """Support indexing to access tokens."""
        if isinstance(key, slice):
            # Return a list of tokens for slices
            return self.tokens[key]
        return self.tokens[key]

    def __iter__(self) -> Iterator[Token]:
        """Make the document iterable over tokens."""
        return iter(self.tokens)

    @property
    def sents(self) -> Iterator[Span]:
        """Return an iterator over sentence spans, similar to spaCy."""
        return iter(self.sentence_spans)


class StanzaDocAdapter:
    """
    Adapter class that converts Stanza documents to a format compatible with 
    our pipeline (mimicking spaCy interface).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
 
    def convert(self, stanza_doc: Any, text: str) -> CompatibleDoc:
        """Convert Stanza document to a format compatible with existing code."""
        doc = CompatibleDoc(text)

        # Extract sentences
        token_idx = 0
        for sent_idx, sent in enumerate(stanza_doc.sentences):
            # Get sentence text
            sent_text = getattr(sent, 'text', "")
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
                sent_start = sum(len(getattr(s, 'text', '')) 
                                for s in stanza_doc.sentences[:sent_idx])
            
            # Track token indices for sentence spans
            token_indices = []
            
            # Extract tokens from current sentence
            token_indices = self._extract_tokens(sent, doc, token_idx)
            if token_indices:
                token_idx = token_indices[-1] + 1
                
                # Create a sentence span with token indices and reference to tokens
                span = Span(token_indices, sent_text, doc.tokens)
                doc.sentence_spans.append(span)
        
        return doc
    
    def _extract_tokens(self, sent, doc, start_idx):
        """Extract tokens from a sentence and add them to the document."""
        token_indices = []
        token_idx = start_idx
        
        if hasattr(sent, 'words'):
            # Process words directly from the sentence
            for word in sent.words:
                token = self._create_token(word, token_idx)
                doc.tokens.append(token)
                doc.words.append(token)
                token_indices.append(token_idx)
                token_idx += 1
                
        elif hasattr(sent, 'tokens'):
            # Process tokens and their words
            for token in sent.tokens:
                if hasattr(token, 'words'):
                    # Handle multi-word tokens
                    for word in token.words:
                        token = self._create_token(word, token_idx)
                        doc.tokens.append(token)
                        doc.words.append(token)
                        token_indices.append(token_idx)
                        token_idx += 1
                else:
                    # Single token
                    token = self._create_token(token, token_idx)
                    doc.tokens.append(token)
                    doc.words.append(token)
                    token_indices.append(token_idx)
                    token_idx += 1
                    
        return token_indices
    
    def _create_token(self, word_or_token, index):
        """Create a Token object from a Stanza word or token."""
        text = getattr(word_or_token, 'text', "")
        
        # Handle different POS tag attributes between Stanza versions
        pos = None
        if hasattr(word_or_token, 'upos'):
            pos = word_or_token.upos  # Universal POS tag
        elif hasattr(word_or_token, 'pos'):
            pos = word_or_token.pos
            
        # Get lemma or default to text
        lemma = getattr(word_or_token, 'lemma', text)
        
        return Token(text=text, pos=pos, lemma=lemma, i=index)


class GreekProcessor:
    """Processor specifically for Ancient Greek using Stanza."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processor = None
        self.adapter = StanzaDocAdapter()
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize the Stanza processor for Ancient Greek."""
        try:
            import stanza
            
            # Download the model if needed (this will only happen once)
            try:
                # Try simple download first - most compatible across versions
                stanza.download('grc')
            except (ImportError, OSError) as e:
                self.logger.warning("Note: %s", e)
                self.logger.warning("Could not download Stanza Greek model - "
                                   "might be offline or already downloaded")
            
            # Initialize the Ancient Greek pipeline with processors
            self.processor = stanza.Pipeline('grc', processors='tokenize,pos,lemma, depparse', download_method=None)
            self.logger.info("Successfully initialized Stanza processor for Ancient Greek")
            self.pipe_names =['tokenize', 'pos', 'lemma', 'depparse']
            
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.error("Error importing Stanza: %s", e)
            self.processor = self._create_dummy_processor()
        except OSError as e:
            self.logger.error("Error loading Stanza models: %s", e)
            self.processor = self._create_dummy_processor()
    
    def _create_dummy_processor(self):
        """Create a simple fallback processor that does minimal processing."""
        self.logger.warning("Using fallback dummy processor for Ancient Greek")
        
        class DummyProcessor:
            def __call__(self, text):
                # Create a minimal Stanza-like document
                class DummyDoc:
                    def __init__(self, text):
                        self.text = text
                        self.sentences = []
                        
                        # Create a dummy sentence with simple tokenization
                        if text.strip():
                            # Split by punctuation for simple sentence splitting
                            import re
                            sentence_texts = re.split(r'[.!?]+', text)
                            
                            for sent_idx, sent_text in enumerate(sentence_texts):
                                if not sent_text.strip():
                                    continue
                                    
                                class DummySentence:
                                    def __init__(self, text, idx):
                                        self.text = text
                                        self.words = []
                                        self.idx = idx
                                        
                                        # Split by spaces for simple tokenization
                                        for word_idx, word_text in enumerate(text.split()):
                                            class DummyWord:
                                                def __init__(self, text, i):
                                                    self.text = text
                                                    self.id = i
                                                    self.upos = "X"  # Unknown POS
                                                    self.lemma = text
                                            
                                            self.words.append(DummyWord(word_text, word_idx))
                                
                                self.sentences.append(DummySentence(sent_text.strip(), sent_idx))
                
                return DummyDoc(text)
        
        return DummyProcessor()
    
    def __call__(self, text):
        """Process text and return a compatible document."""
        try:
            # Process with Stanza
            stanza_doc = self.processor(text)
            # Convert to compatible format
            return self.adapter.convert(stanza_doc, text)
        except (AttributeError, ValueError, TypeError) as e:
            self.logger.error("Error processing text with Stanza: %s", e)
            # Return a minimal document with the text
            return CompatibleDoc(text)


def get_stanza_greek_processor():
    """
    Create a processor for Ancient Greek that integrates with the existing pipeline.
    
    Returns:
        GreekProcessor: A callable processor that returns compatible document objects
    """
    return GreekProcessor()