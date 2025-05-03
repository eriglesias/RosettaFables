# src/aesop_spacy/io/serializer.py
from typing import Any, Dict, List, Set, Optional, Union
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
            except Exception as e:
                self.logger.warning(f"Error calling to_dict() on {type(obj).__name__}: {e}")
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
            except Exception as e:
                self.logger.warning(f"Error converting {type(obj).__name__} to string: {e}")
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
                result['tokens'] = [self.serialize_token(token, visited) 
                                   for token in obj]
                result['ents'] = [self.serialize_span(ent, visited) 
                                 for ent in obj.ents]
        
        except Exception as e:
            self.logger.warning(f"Error serializing {type(obj).__name__}: {e}")
            result['error'] = str(e)
        
        return result
    
    def serialize_token(self, token: Any, visited: Set[int]) -> Dict[str, Any]:
        """
        Serialize a spaCy Token object.
        
        Args:
            token: A spaCy Token
            visited: Set of visited object IDs
            
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
        serialized = {}
        
        # Copy basic sentence fields
        for key in ['text', 'start', 'end']:
            if key in sentence_data:
                serialized[key] = sentence_data[key]
        
        # Handle tokens and POS tags
        if 'tokens' in sentence_data:
            serialized['tokens'] = self.serialize(sentence_data['tokens'])
        if 'pos_tags' in sentence_data:
            serialized['pos_tags'] = self.serialize(sentence_data['pos_tags'])
        
        # Special handling for dependency structure
        if 'dependencies' in sentence_data:
            self.logger.debug(f"Serializing {len(sentence_data['dependencies'])} dependencies for sentence")
            serialized['dependencies'] = []
            for dep in sentence_data['dependencies']:
                if isinstance(dep, dict):
                    serialized_dep = {
                        'dep': dep.get('dep', ''),
                        'head_id': dep.get('head_id'),
                        'dependent_id': dep.get('dependent_id'),
                        'head_text': dep.get('head_text', ''),
                        'dependent_text': dep.get('dependent_text', '')
                    }
                    serialized['dependencies'].append(serialized_dep)
        
        # Handle root information
        if 'root' in sentence_data:
            serialized['root'] = self.serialize(sentence_data['root'])
        
        return serialized