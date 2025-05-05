#serializer.py
"""
Serializer module for converting spaCy objects to JSON-serializable structures.

This module provides functionality to convert spaCy documents, tokens, spans
and other objects into JSON-serializable dictionaries, with special handling
for nested structures, circular references, and dependency information.
"""

from typing import Any, Dict, Set, Optional
import logging


class SpacySerializer:
    """Converts spaCy objects to JSON-serializable data structures."""
    
    def __init__(self):
        """Initialize the serializer."""
        self.logger = logging.getLogger(__name__)
    
    def serialize(self, obj: Any, visited: Optional[Set[int]] = None) -> Any:
        """
        Recursively serialize a spaCy object or any Python object to a JSON-compatible representation.
        
        Args:
            obj: The object to serialize
            visited: Set of object IDs already visited (for cycle detection)
            
        Returns:
            JSON-serializable representation of the object
        """
        # Initialize the visited set on the first call
        if visited is None:
            visited = set()
        
        # Get the object ID to detect cycles
        obj_id = id(obj)
        
        # Handle circular references
        if obj_id in visited:
            return "<circular reference>"
        
        # Add this object to the visited set to prevent cycles
        visited.add(obj_id)
        
        # Special handling for fables with sentences
        if isinstance(obj, dict) and 'sentences' in obj:
            self.logger.debug("Serializing a fable with %d sentences", len(obj['sentences']))
            
            # Check for document-level dependencies
            if 'dependencies' in obj:
                self.logger.debug("Fable has %d document-level dependencies", len(obj['dependencies']))
            
            result = {}
            for k, v in obj.items():
                if k != 'sentences':
                    result[k] = self.serialize(v, visited.copy())
            
            # Explicitly use serialize_sentence for each sentence
            result['sentences'] = []
            for i, sent in enumerate(obj['sentences']):
                # Log before serialization to see if dependencies exist
                if 'dependencies' in sent:
                    self.logger.debug("Sentence %d has %d dependencies before serialization", 
                                     i, len(sent['dependencies']))
                else:
                    self.logger.debug("Sentence %d has no dependencies before serialization", i)
                    
                serialized_sent = self.serialize_sentence(sent)
                
                # Verify dependencies were preserved
                if 'dependencies' in sent and not serialized_sent.get('dependencies'):
                    self.logger.warning("Dependencies lost during serialization for sentence %d", i)
                    # Try to salvage the dependencies by directly copying
                    if isinstance(sent['dependencies'], list):
                        serialized_sent['dependencies'] = []
                        for dep in sent['dependencies']:
                            if isinstance(dep, dict):
                                serialized_sent['dependencies'].append(dep.copy())
                            else:
                                serialized_sent['dependencies'].append(self.serialize(dep, visited.copy()))
                
                result['sentences'].append(serialized_sent)
            
            # If we have document-level dependencies, make sure they're preserved
            if 'dependencies' in obj:
                # Directly preserve dependencies to avoid any potential serialization issues
                if isinstance(obj['dependencies'], list):
                    result['dependencies'] = []
                    for dep in obj['dependencies']:
                        if isinstance(dep, dict):
                            result['dependencies'].append(dep.copy())
                        else:
                            result['dependencies'].append(self.serialize(dep, visited.copy()))
                    self.logger.debug("Preserved %d document-level dependencies", 
                                     len(result['dependencies']))
                else:
                    result['dependencies'] = self.serialize(obj['dependencies'], visited.copy())
            
            return result
        
        # Handle basic types
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        
        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            return [self.serialize(item, visited.copy()) for item in obj]
        
        # Handle dictionaries
        elif isinstance(obj, dict):
            return {str(key): self.serialize(value, visited.copy()) 
                   for key, value in obj.items()}
        
        # Handle spaCy Doc, Token, Span objects and others with to_dict method
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            try:
                dict_repr = obj.to_dict()
                return self.serialize(dict_repr, visited.copy())
            except AttributeError as e:
                self.logger.warning("Error calling to_dict() on %s: %s", type(obj).__name__, e)
                return str(obj)
            except ValueError as e:
                self.logger.warning("Value error in to_dict() on %s: %s", type(obj).__name__, e)
                return str(obj)
        
        # Handle objects with __dict__ (custom classes)
        elif hasattr(obj, '__dict__'):
            # Filter out private attributes (those starting with _)
            filtered_dict = {k: v for k, v in obj.__dict__.items() 
                            if not k.startswith('_')}
            return self.serialize(filtered_dict, visited.copy())

        # For spaCy-specific objects without to_dict
        elif hasattr(obj, 'text'):
            # This catches most spaCy objects like Token, Span
            return self.serialize_spacy_object(obj, visited)

        # For anything else, convert to string
        else:
            try:
                return str(obj)
            except ValueError as e:
                self.logger.warning("Error converting %s to string: %s", type(obj).__name__, e)
                return "<unserializable object>"

    def serialize_spacy_object(self, obj: Any, visited: Set[int]) -> Dict[str, Any]:
        """
        Serialize a spaCy object that doesn't have a to_dict method.
        
        Args:
            obj: A spaCy object
            visited: Set of visited object IDs
            
        Returns:
            Dictionary representation of the object
        """
        result = {}

        # Common attributes to extract from spaCy objects
        try:
            if hasattr(obj, 'text'):
                result['text'] = obj.text

            if hasattr(obj, 'lemma_'):
                result['lemma'] = obj.lemma_

            if hasattr(obj, 'pos_'):
                result['pos'] = obj.pos_

            if hasattr(obj, 'tag_'):
                result['tag'] = obj.tag_

            if hasattr(obj, 'dep_'):
                result['dep'] = obj.dep_

            if hasattr(obj, 'ent_type_'):
                result['ent_type'] = obj.ent_type_
 
            if hasattr(obj, 'ent_iob_'):
                result['ent_iob'] = obj.ent_iob_
                
            if hasattr(obj, 'start'):
                result['start'] = obj.start
                
            if hasattr(obj, 'end'):
                result['end'] = obj.end
                
            if hasattr(obj, 'i'):
                result['i'] = obj.i
            
            # For Doc objects, serialize tokens
            if hasattr(obj, 'ents') and hasattr(obj, '__iter__'):
                # This is likely a Doc object
                result['tokens'] = [self.serialize_token(token) for token in obj]
                result['ents'] = [self.serialize_span(ent, visited) for ent in obj.ents]
        
        except AttributeError as e:
            self.logger.warning("Attribute error serializing %s: %s", type(obj).__name__, e)
            result['error'] = str(e)
        except ValueError as e:
            self.logger.warning("Value error serializing %s: %s", type(obj).__name__, e)
            result['error'] = str(e)
        
        return result
    
    def serialize_token(self, token: Any) -> Dict[str, Any]:
        """
        Serialize a spaCy Token object.
        
        Args:
            token: A spaCy Token
            
        Returns:
            Dictionary representation of the token
        """
        return {
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'dep': token.dep_,
            'is_stop': token.is_stop,
            'i': token.i,
        }
    
    def serialize_span(self, span: Any, visited: Set[int]) -> Dict[str, Any]:
        """
        Serialize a spaCy Span object.
        
        Args:
            span: A spaCy Span
            visited: Set of visited object IDs
            
        Returns:
            Dictionary representation of the span
        """
        return {
            'text': span.text,
            'start': span.start,
            'end': span.end,
            'label': span.label_,
        }
    
    def serialize_sentence(self, sentence_data):
        """
        Serialize a sentence dictionary with special handling for dependencies.
        """
        # Start with a debug log to see what we're working with
        self.logger.debug("Serializing sentence: %s...", sentence_data.get('text', '')[:30])
        
        serialized = {}
        
        # Check if we have dependencies to serialize
        has_dependencies = 'dependencies' in sentence_data and sentence_data['dependencies']
        if has_dependencies:
            self.logger.debug("Found %d dependencies in sentence", len(sentence_data['dependencies']))
            
            # Log the first dependency to help with debugging
            if len(sentence_data['dependencies']) > 0:
                self.logger.debug("Sample dependency: %s", sentence_data['dependencies'][0])
        else:
            self.logger.debug("No dependencies found in sentence to serialize")
        
        # Copy basic sentence fields
        for key in ['text', 'start', 'end']:
            if key in sentence_data:
                serialized[key] = sentence_data[key]
        
        # Handle tokens and POS tags
        if 'tokens' in sentence_data:
            serialized['tokens'] = self.serialize(sentence_data['tokens'])
        if 'pos_tags' in sentence_data:
            serialized['pos_tags'] = self.serialize(sentence_data['pos_tags'])
        
        # Special handling for dependency structure - more defensive
        serialized['dependencies'] = []  # Always initialize the key
        
        if has_dependencies:
            for dep in sentence_data['dependencies']:
                try:
                    if isinstance(dep, dict):
                        # Create a copy to preserve the original structure
                        serialized_dep = dep.copy()
                        
                        # Ensure all required fields have at least default values
                        if 'dep' not in serialized_dep:
                            serialized_dep['dep'] = ''
                        if 'head_id' not in serialized_dep:
                            serialized_dep['head_id'] = -1
                        if 'dependent_id' not in serialized_dep:
                            serialized_dep['dependent_id'] = -1
                        if 'head_text' not in serialized_dep:
                            serialized_dep['head_text'] = ''
                        if 'dependent_text' not in serialized_dep:
                            serialized_dep['dependent_text'] = ''
                            
                        serialized['dependencies'].append(serialized_dep)
                    else:
                        self.logger.warning("Unexpected dependency format: %s", type(dep))
                        # Try to salvage what we can - convert to dict if possible
                        if hasattr(dep, '__dict__'):
                            # Convert object to dictionary
                            dep_dict = {k: v for k, v in dep.__dict__.items() 
                                       if not k.startswith('_')}
                            serialized['dependencies'].append(dep_dict)
                        else:
                            # Fall back to serialization
                            serialized['dependencies'].append(self.serialize(dep))
                except (AttributeError, ValueError, TypeError) as e:
                    self.logger.error("Error serializing dependency: %s", e)
                    # Continue with other dependencies rather than failing completely
            
            # Verify dependencies were preserved
            if not serialized.get('dependencies'):
                self.logger.warning("Dependencies were lost during serialization!")
            else:
                self.logger.debug("Successfully serialized %d dependencies", 
                                 len(serialized['dependencies']))
        
        # Handle root information
        if 'root' in sentence_data:
            serialized['root'] = self.serialize(sentence_data['root'])
        
        return serialized