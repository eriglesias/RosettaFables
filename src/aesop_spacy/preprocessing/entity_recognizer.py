from typing import List, Dict, Any, Optional
import logging

class EntityRecognizer:
    """
    Handle custom entity recognition for fables.
    Follows the Strategy pattern to allow different recognition strategies.
    """
    
    def __init__(self):
        """Initialize the entity recognizer."""
        self.logger = logging.getLogger(__name__)
    
    def add_entity_patterns(self, nlp, language: str) -> None:
        """
        Add custom entity patterns to a spaCy model.
        
        Args:
            nlp: The spaCy model to enhance
            language: Language code to determine appropriate patterns
        """
        # Skip if model doesn't support entity rulers or is wrong language
        if not hasattr(nlp, 'add_pipe'):
            self.logger.warning(f"Model doesn't support adding pipes, skipping entity patterns")
            return
            
        # Get patterns for this language
        patterns = self.get_entity_patterns(language)
        if not patterns:
            return
            
        # Check if an entity ruler already exists
        if 'animal_ruler' not in nlp.pipe_names:
            try:
                # Add entity ruler
                ruler = nlp.add_pipe("entity_ruler", before="ner", name="animal_ruler")
                ruler.add_patterns(patterns)
                
                # Set to overwrite existing entities
                if hasattr(ruler, 'overwrite'):
                    ruler.overwrite = True
                    
                self.logger.info(f"Added {len(patterns)} entity patterns for {language}")
            except Exception as e:
                self.logger.error(f"Error adding entity ruler: {e}")
    
    def get_entity_patterns(self, language: str) -> List[Dict[str, Any]]:
        """
        Get custom entity patterns for a specific language.
        
        Args:
            language: ISO language code
            
        Returns:
            List of pattern dictionaries compatible with spaCy's EntityRuler
        """
        patterns = []
        
        # English patterns
        if language == 'en':
            animals = ["wolf", "lamb", "fox", "lion", "mouse", "crane", "goat", 
                      "mosquito", "frog", "cat", "dog", "crow"]
            
            # Add single token patterns
            for animal in animals:
                patterns.append({"label": "ANIMAL_CHAR", "pattern": animal})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": animal.capitalize()})
                
                # Add with determiners
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "the"}, {"LOWER": animal}]})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "a"}, {"LOWER": animal}]})
        
        # Dutch patterns
        elif language == 'nl':
            animals = ["wolf", "lam", "vos", "leeuw", "muis", "kraanvogel", "geit", "mug"]
            # Similar pattern creation as English
            for animal in animals:
                patterns.append({"label": "ANIMAL_CHAR", "pattern": animal})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": animal.capitalize()})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "de"}, {"LOWER": animal}]})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "een"}, {"LOWER": animal}]})
        
        # German patterns
        elif language == 'de':
            animals = ["Wolf", "Lamm", "Fuchs", "Löwe", "Maus", "Kranich", "Ziege", "Mücke"]
            # Similar pattern creation as English
            for animal in animals:
                patterns.append({"label": "ANIMAL_CHAR", "pattern": animal})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": animal.lower()})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "der"}, {"LOWER": animal.lower()}]})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "die"}, {"LOWER": animal.lower()}]})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "das"}, {"LOWER": animal.lower()}]})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "ein"}, {"LOWER": animal.lower()}]})
                patterns.append({"label": "ANIMAL_CHAR", "pattern": [{"LOWER": "eine"}, {"LOWER": animal.lower()}]})
        
        # Spanish patterns - add if needed
        
        return patterns