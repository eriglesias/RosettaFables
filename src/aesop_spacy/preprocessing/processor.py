#processor.py
"""processes"""
from typing import Dict, List, Any, Tuple
import logging


class FableProcessor:
    """
    Process fables with NLP models to extract linguistic features.
    """

    def __init__(self):
        """Initialize the fable processor."""
        self.logger = logging.getLogger(__name__)

    def process_fable(self, fable: Dict[str, Any], nlp_model) -> Dict[str, Any]:
        """
        Process a fable with spaCy to extract linguistic features.
        
        Args:
            fable: Dictionary containing fable data
            nlp_model: Loaded spaCy model
            
        Returns:
            Fable with added linguistic features
        """
        # Create a copy to avoid modifying the original
        processed = fable.copy()

        # Use the extracted body if available, otherwise use the regular body
        body = fable.get('extracted_body', fable.get('body', ''))

        if not body:
            self.logger.warning("No content to process for fable: %s", fable.get('title'))
            processed.update({
                'doc_length': 0,
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'entities': [],
                'sentences': [],
                'dependencies': [],  
            })
            return processed

        # Apply the NLP model
        doc = nlp_model(body)

        # Check if model has parser component
        if hasattr(nlp_model, 'pipe_names'):
            self.logger.info("Model pipeline components: %s", nlp_model.pipe_names)
        if 'parser' not in nlp_model.pipe_names:
            self.logger.warning("Model %s does not have parser component!", nlp_model)

        # Extract dependencies before serialization and log them
        sentences = self._extract_sentences(doc)
        total_deps = sum(len(s.get('dependencies', [])) for s in sentences)
        self.logger.info("Extracted %d dependencies across %d sentences", total_deps, len(sentences))

        # Get document-level dependencies
        doc_dependencies = self._extract_dependencies(doc)
        
        # Extract linguistic features
        processed.update({
            'doc_length': len(doc),
            'tokens': self._extract_tokens(doc),
            'lemmas': self._extract_lemmas(doc),
            'pos_tags': self._extract_pos_tags(doc),
            'entities': self._extract_entities(doc),
            'sentences': self._extract_sentences(doc),
            'dependencies': doc_dependencies  # Add document-level dependencies
        })

        # Log dependency stats
        self.logger.info("Extracted %d document-level dependencies", len(doc_dependencies))
        sentence_deps = sum(len(s.get('dependencies', [])) for s in processed['sentences'])
        self.logger.info("Extracted %d sentence-level dependencies across %d sentences", 
                         sentence_deps, len(processed['sentences']))

        # Process moral if present
        if 'extracted_moral' in fable and fable['extracted_moral'].get('text'):
            moral_text = fable['extracted_moral']['text']
            processed['moral_analysis'] = self._process_moral(moral_text, nlp_model)

        return processed

    def _extract_tokens(self, doc) -> List[Tuple[str, int]]:
        """Extract tokens with their positions."""
        return [(token.text, token.i) for token in doc]

    def _extract_lemmas(self, doc) -> List[Tuple[str, int]]:
        """Extract lemmas with their positions."""
        return [(token.lemma_, token.i) for token in doc]

    def _extract_pos_tags(self, doc) -> List[Tuple[str, str]]:
        """Extract part-of-speech tags."""
        return [(token.text, token.pos_) for token in doc]

    def _extract_entities(self, doc) -> List[Tuple[str, str, int, int]]:
        """Extract named entities with their types and positions."""
        return [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]

    def _extract_sentences(self, doc) -> List[Dict[str, Any]]:
        """Extract sentences with their metadata and dependencies."""
        sentences = []

        # Check if this is a spaCy doc or a Stanza doc
        is_stanza = hasattr(doc, 'sentences') and not hasattr(doc, 'sents')
        if is_stanza:
            # Handle Stanza document
            self.logger.debug("Processing Stanza document with %d sentences", len(doc.sentences))
            for sent in doc.sentences:
                # Extract dependencies for Stanza
                dependencies = []
                try:
                    for word in sent.words:
                        if word.head > 0:  # Skip root (head=0)
                            dependencies.append({
                                'dep': word.deprel,
                                'head_id': word.head,
                                'dependent_id': word.id,
                                'head_text': sent.words[word.head-1].text if word.head <= len(sent.words) else '',
                                'dependent_text': word.text
                            })
                    
                    self.logger.debug("Extracted %d dependencies for Stanza sentence", len(dependencies))
                except Exception as e:
                    self.logger.warning("Failed to extract Stanza dependencies: %s", e)
                    dependencies = []
                
                sentence_data = {
                    'text': sent.text,
                    'start': sent.tokens[0].id if sent.tokens else 0,
                    'end': sent.tokens[-1].id if sent.tokens else 0,
                    'dependencies': dependencies
                }
                sentences.append(sentence_data)
        else:
            # Handle spaCy document with improved error handling
            self.logger.debug("Processing spaCy document")
            try:
                for sent in doc.sents:
                    try:
                        # Extract dependencies for spaCy
                        dependencies = []
                        for token in sent:
                            if token.dep_ != '':  # Skip tokens without dependency relation
                                dependencies.append({
                                    'dep': token.dep_,
                                    'head_id': token.head.i,
                                    'dependent_id': token.i,
                                    'head_text': token.head.text,
                                    'dependent_text': token.text
                                })
                        
                        self.logger.debug("Found %d dependencies in sentence", len(dependencies))
                        
                        # Some spans might not have root attribute
                        root_data = {}
                        if hasattr(sent, 'root'):
                            root_data = {
                                'text': sent.root.text,
                                'pos': sent.root.pos_,
                                'dep': sent.root.dep_
                            }
                        else:
                            root_data = {'text': "", 'pos': "", 'dep': ""}
                    except AttributeError as e:
                        # Fallback if there's any issue
                        self.logger.warning("Error extracting dependencies for sentence: %s", e)
                        root_data = {'text': "", 'pos': "", 'dep': ""}
                        dependencies = []
                    
                    sentence_data = {
                        'text': sent.text,
                        'start': sent.start,
                        'end': sent.end,
                        'root': root_data,
                        'dependencies': dependencies
                    }
                    sentences.append(sentence_data)
            except Exception as e:
                self.logger.warning("Error extracting sentences from spaCy doc: %s", e)
        
        # Log dependency stats
        total_deps = sum(len(s.get('dependencies', [])) for s in sentences)
        self.logger.info("Extracted %d sentences with %d total dependencies", len(sentences), total_deps)
        
        return sentences

    def _process_moral(self, moral_text: str, nlp_model) -> Dict[str, Any]:
        """Process the moral text to extract linguistic features."""
        if not moral_text:
            return {}

        moral_doc = nlp_model(moral_text)

        return {
            'tokens': [token.text for token in moral_doc],
            'lemmas': [token.lemma_ for token in moral_doc],
            'pos_tags': [(token.text, token.pos_) for token in moral_doc],
            'entities': [(ent.text, ent.label_) for ent in moral_doc.ents],
            'keywords': self._extract_keywords(moral_doc)
        }

    def _extract_keywords(self, doc) -> List[str]:
        """Extract important keywords from text based on POS tags and other features."""
        keywords = []

        # Prioritize nouns, verbs, and adjectives that aren't stopwords
        for token in doc:
            if (not token.is_stop and token.is_alpha and 
                token.pos_ in ('NOUN', 'VERB', 'ADJ', 'PROPN') and
                len(token.text) > 2):
                keywords.append(token.text.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]

        return unique_keywords
    
    def _extract_dependencies(self, doc) -> List[Dict[str, Any]]:
        """Extract all dependency relations from the document."""
        dependencies = []
        
        try:
            # For spaCy doc
            if hasattr(doc, 'has_annotation') and doc.has_annotation('DEP'):
                for token in doc:
                    if token.dep_ and token.dep_ != '':
                        dependencies.append({
                            'dep': token.dep_,
                            'head_id': token.head.i,
                            'dependent_id': token.i,
                            'head_text': token.head.text,
                            'dependent_text': token.text
                        })
            # For Stanza doc
            elif hasattr(doc, 'sentences'):
                for sent in doc.sentences:
                    for word in sent.words:
                        if word.head > 0:  # Skip root (head=0)
                            dependencies.append({
                                'dep': word.deprel,
                                'head_id': word.head,
                                'dependent_id': word.id,
                                'head_text': sent.words[word.head-1].text if word.head <= len(sent.words) else '',
                                'dependent_text': word.text
                            })
            
            self.logger.info("Extracted %d document-level dependencies", len(dependencies))
        except Exception as e:
            self.logger.warning("Error extracting dependencies: %s", e)
            dependencies = []
        
        return dependencies