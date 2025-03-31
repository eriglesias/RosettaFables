"""
Entity recognition utilities for fable text analysis.

This module provides custom entity recognition functionality for fables,
with special handling for animal characters across multiple languages.
It allows for consistent identification of characters despite capitalization
variations and supports proper canonical forms to improve entity tracking.

The EntityRecognizer follows the Strategy pattern to allow different
recognition strategies based on language and text characteristics.
"""

from typing import List, Dict, Any, Optional
import logging
from spacy.language import Language


class EntityRecognizer:
    """
    Handle custom entity recognition for fables.
    Follows the Strategy pattern to allow different recognition strategies.
    """

    def __init__(self):
        """Initialize the entity recognizer."""
        self.logger = logging.getLogger(__name__)
        # Track recognized entities across documents
        self.recognized_entities = {}

    def add_entity_patterns(self, nlp, language: str, canonical_forms: Dict[str, str] = None) -> None:
        """
        Add custom entity patterns to a spaCy model.
        
        Args:
            nlp: The spaCy model to enhance
            language: Language code to determine appropriate patterns
            canonical_forms: Optional dictionary of canonical forms for consistent recognition
        """
        # Skip if model doesn't support entity rulers or is wrong language
        if not hasattr(nlp, 'add_pipe'):
            self.logger.warning("No entity patterns available for language: %s", language)
            return

        # Get patterns for this language
        patterns = self.get_entity_patterns(language, canonical_forms)
        if not patterns:
            self.logger.info("Added %d entity patterns for %s", len(patterns), language)
            return

        try:
            # Check if an entity ruler already exists
            if 'animal_ruler' not in nlp.pipe_names:
                # Add entity ruler before NER to influence its decisions
                ruler = nlp.add_pipe("entity_ruler", before="ner", name="animal_ruler")
                ruler.add_patterns(patterns)

                # Set to overwrite existing entities for consistent recognition
                if hasattr(ruler, 'overwrite'):
                    ruler.overwrite = True 
                self.logger.info(f"Added {len(patterns)} entity patterns for {language}")
            else:
                # Update existing ruler with new patterns
                ruler = nlp.get_pipe("animal_ruler")
                ruler.add_patterns(patterns)
                self.logger.info(f"Updated entity ruler with {len(patterns)} patterns")
                
        except ValueError as e:
            # Handle the case where "ner" component doesn't exist
            self.logger.warning(f"Could not add entity ruler before 'ner': {e}")
            try:
                # Try adding at the end of the pipeline instead
                ruler = nlp.add_pipe("entity_ruler", name="animal_ruler")
                ruler.add_patterns(patterns)
                self.logger.info(f"Added entity ruler at end of pipeline with {len(patterns)} patterns")
            except Exception as e2:
                self.logger.error(f"Failed to add entity ruler: {e2}")
        except Exception as e:
            self.logger.error(f"Error adding entity ruler: {e}")

    def get_entity_patterns(self, language: str, canonical_forms: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Get custom entity patterns for a specific language.
        
        Args:
            language: ISO language code
            canonical_forms: Optional dictionary of canonical forms for consistent recognition
            
        Returns:
            List of pattern dictionaries compatible with spaCy's EntityRuler
        """
        # Get the base animals for this language
        animals = self._get_animal_list(language)
        
        # Get the appropriate determiners for this language
        determiners = self._get_determiners(language)
        
        # Create patterns based on language
        if language in ['en', 'nl', 'de', 'es']:
            # Use helper method to create consistent patterns
            patterns = self._create_animal_patterns(
                animals, 
                determiners, 
                language, 
                canonical_forms
            )
            return patterns
        else:
            self.logger.warning(f"Language not supported for entity patterns: {language}")
            return []

    def _create_animal_patterns(self, 
                               animals: List[str], 
                               determiners: List[str],
                               language: str,
                               canonical_forms: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Create entity patterns for animal characters with consistent formatting.
        
        Args:
            animals: List of animal names
            determiners: List of determiners (articles) for the language
            language: Language code for language-specific handling
            canonical_forms: Optional dictionary of canonical forms
            
        Returns:
            List of pattern dictionaries for the entity ruler
        """
        patterns = []
        
        # Use canonical forms if provided
        if canonical_forms:
            preferred_forms = canonical_forms
        else:
            # Otherwise create default forms
            preferred_forms = {animal.lower(): animal.capitalize() for animal in animals}
        
        for animal in animals:
            animal_lower = animal.lower()
            
            # Get the preferred form of this animal
            preferred = preferred_forms.get(animal_lower, animal.capitalize())
            
            # Add base forms
            patterns.append({
                "label": "ANIMAL_CHAR", 
                "pattern": preferred,
                "id": f"animal_{animal_lower}"
            })
            
            # Also match lowercase form
            if preferred != animal_lower:
                patterns.append({
                    "label": "ANIMAL_CHAR", 
                    "pattern": animal_lower,
                    "id": f"animal_{animal_lower}"
                })
            
            # Add capitalized form if it's different
            capitalized = animal_lower.capitalize()
            if capitalized != preferred and capitalized != animal_lower:
                patterns.append({
                    "label": "ANIMAL_CHAR", 
                    "pattern": capitalized,
                    "id": f"animal_{animal_lower}"
                })
            
            # Add forms with determiners
            for det in determiners:
                # Add lowercase determiner + animal
                patterns.append({
                    "label": "ANIMAL_CHAR", 
                    "pattern": [{"LOWER": det.lower()}, {"LOWER": animal_lower}],
                    "id": f"animal_{animal_lower}_with_det"
                })
                
                # Add capitalized determiner + animal
                patterns.append({
                    "label": "ANIMAL_CHAR", 
                    "pattern": [{"TEXT": det.capitalize()}, {"LOWER": animal_lower}],
                    "id": f"animal_{animal_lower}_with_det_cap"
                })
            
            # Add variations for German where all nouns are capitalized
            if language == 'de':
                patterns.append({
                    "label": "ANIMAL_CHAR", 
                    "pattern": animal_lower.capitalize(),
                    "id": f"animal_{animal_lower}_cap"
                })
            
            # Add plural forms
            plural = self._get_plural_form(animal_lower, language)
            if plural:
                patterns.append({
                    "label": "ANIMAL_CHAR", 
                    "pattern": plural,
                    "id": f"animal_{animal_lower}_plural"
                })
                
                # Add capitalized plural
                patterns.append({
                    "label": "ANIMAL_CHAR", 
                    "pattern": plural.capitalize(),
                    "id": f"animal_{animal_lower}_plural_cap"
                })
        
        return patterns

    def _get_animal_list(self, language: str) -> List[str]:
        """
        Get a list of common animal characters for a language.
        
        Args:
            language: ISO language code
            
        Returns:
            List of animal names
        """
        # Define animals by language
        animals_by_language = {
            'en': ["wolf", "lamb", "fox", "lion", "mouse", "crane", "goat",
                  "mosquito", "frog", "cat", "dog", "crow", "city mouse", "country mouse"],
            'nl': ["wolf", "lam", "vos", "leeuw", "muis", "kraanvogel", "geit", 
                  "mug", "kikker", "kat", "hond", "kraai", "stadsmuis", "veldmuis", "geitje"],
            'de': ["wolf", "lamm", "fuchs", "löwe", "maus", "kranich", "ziege", 
                  "mücke", "frosch", "katze", "hund", "krähe", "stadtmaus", "landmaus"],
            'es': ["lobo", "cordero", "zorro", "león", "ratón", "grulla", "cabra", 
                  "mosquito", "rana", "gato", "perro", "cuervo", "ratón de ciudad", "ratón de campo"],
        }
        
        return animals_by_language.get(language, [])

    def _get_determiners(self, language: str) -> List[str]:
        """
        Get a list of determiners (articles) for a language.
        
        Args:
            language: ISO language code
            
        Returns:
            List of determiners
        """
        # Define determiners by language
        determiners_by_language = {
            'en': ["the", "a", "an"],
            'nl': ["de", "het", "een"],
            'de': ["der", "die", "das", "ein", "eine"],
            'es': ["el", "la", "un", "una", "los", "las", "unos", "unas"],
        }
        
        return determiners_by_language.get(language, [])

    def _get_plural_form(self, animal: str, language: str) -> Optional[str]:
        """
        Get the plural form of an animal name based on language rules.
        
        Args:
            animal: Singular animal name
            language: ISO language code
            
        Returns:
            Plural form or None if unavailable
        """
        # Custom plural mappings (for irregular plurals)
        custom_plurals = {
            'en': {
                'wolf': 'wolves',
                'fox': 'foxes',
                'mouse': 'mice',
                'goose': 'geese',
                'city mouse': 'city mice',
                'country mouse': 'country mice',
            },
            'nl': {
                'wolf': 'wolven',
                'vos': 'vossen',
                'muis': 'muizen',
                'leeuw': 'leeuwen',
                'geit': 'geiten',
                'geitje': 'geitjes',
                'stadsmuis': 'stadsmuizen',
                'veldmuis': 'veldmuizen',
            },
            'de': {
                'wolf': 'wölfe',
                'fuchs': 'füchse',
                'maus': 'mäuse',
                'stadtmaus': 'stadtmäuse',
                'landmaus': 'landmäuse',
            },
            'es': {
                'lobo': 'lobos',
                'zorro': 'zorros',
                'ratón': 'ratones',
                'león': 'leones',
                'ratón de ciudad': 'ratones de ciudad',
                'ratón de campo': 'ratones de campo',
            }
        }
        
        # Check if we have a custom plural for this animal
        if language in custom_plurals and animal in custom_plurals[language]:
            return custom_plurals[language][animal]
        
        # Apply default language rules for plurals
        if language == 'en':
            if animal.endswith(('s', 'x', 'z', 'ch', 'sh')):
                return animal + 'es'
            elif animal.endswith('y') and not animal.endswith(('ay', 'ey', 'iy', 'oy', 'uy')):
                return animal[:-1] + 'ies'
            else:
                return animal + 's'
        elif language == 'nl':
            # Very simplified Dutch pluralization
            return animal + 'en'
        elif language == 'de':
            # German pluralization is complex, we'd need a more robust approach
            return None
        elif language == 'es':
            # Simplified Spanish pluralization
            if animal.endswith(('s', 'x', 'z')):
                return animal
            elif animal.endswith(('a', 'e', 'i', 'o', 'u')):
                return animal + 's'
            else:
                return animal + 'es'
        
        return None

    def add_character_consolidation(self, nlp) -> None:
        """
        Add a custom component to consolidate character entities.
        
        Args:
            nlp: The spaCy model to enhance
        """
        # Define the component function
        @Language.component("character_consolidator")
        def character_consolidator(doc):
            """Merge character mentions that differ only in capitalization."""
            if not doc.ents:
                return doc
                
            entities = list(doc.ents)
            consolidated_entities = []
            seen_lowercase = {}
            
            # Group entities by lowercase text
            for ent in entities:
                # Only process animal characters
                if ent.label_ != "ANIMAL_CHAR":
                    consolidated_entities.append(ent)
                    continue
                    
                lowercase_text = ent.text.lower()
                if lowercase_text in seen_lowercase:
                    # Skip this entity as we'll handle it with the canonical version
                    continue
                
                # Find all entities with the same lowercase text
                matching_ents = [e for e in entities if e.label_ == "ANIMAL_CHAR" and e.text.lower() == lowercase_text]
                
                # Choose the canonical form (prefer capitalized)
                canonical = max(matching_ents, key=lambda e: int(e.text[0].isupper()))
                seen_lowercase[lowercase_text] = canonical
                consolidated_entities.append(canonical)
            
            # Set the new entities
            doc.ents = tuple(consolidated_entities)
            return doc
        
        try:
            # Add the component after NER
            if "character_consolidator" not in nlp.pipe_names:
                nlp.add_pipe("character_consolidator", after="ner")
                self.logger.info("Added character consolidation component")
        except ValueError as e:
            self.logger.error("ValueError adding consolidation component: %s", e)
        except Exception as e:
            self.logger.error("Unexpected error adding component: %s", e)
            self.logger.debug("Exception type: %s", type(e).__name__)

    def track_entity(self, entity_text: str, entity_label: str, document_id: str) -> None:
        """
        Track recognized entities across documents.
        
        Args:
            entity_text: The text of the recognized entity
            entity_label: The label of the entity (e.g., "ANIMAL_CHAR")
            document_id: Identifier for the document
        """
        # Initialize if needed
        if entity_label not in self.recognized_entities:
            self.recognized_entities[entity_label] = {}
            
        entity_lower = entity_text.lower()
        
        # Initialize this entity if needed
        if entity_lower not in self.recognized_entities[entity_label]:
            self.recognized_entities[entity_label][entity_lower] = {
                "canonical_form": entity_text,
                "mentions": 0,
                "documents": set()
            }
        
        # Update the statistics
        self.recognized_entities[entity_label][entity_lower]["mentions"] += 1
        self.recognized_entities[entity_label][entity_lower]["documents"].add(document_id)
        
        # Prefer capitalized forms for the canonical form
        if entity_text[0].isupper() and not self.recognized_entities[entity_label][entity_lower]["canonical_form"][0].isupper():
            self.recognized_entities[entity_label][entity_lower]["canonical_form"] = entity_text
            
    def get_entity_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recognized entities.
        
        Returns:
            Dictionary with entity statistics
        """
        stats = {}
        
        for label, entities in self.recognized_entities.items():
            stats[label] = {}
            
            for _, data in entities.items():
                stats[label][data["canonical_form"]] = {
                    "mentions": data["mentions"],
                    "document_count": len(data["documents"])
                }

        return stats