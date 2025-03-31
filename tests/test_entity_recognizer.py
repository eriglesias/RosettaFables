# tests/test_entity_recognizer.py
"""
Tests for the EntityRecognizer component.

This module contains tests for the EntityRecognizer class which is responsible
for adding custom entity recognition capabilities to spaCy models for animal
characters in multilingual fables.

The tests use lightweight mocks of spaCy components to verify behavior without
requiring the full spaCy library.
"""
import pytest
from aesop_spacy.preprocessing.entity_recognizer import EntityRecognizer


class DummyRuler:
    """
    Mock implementation of spaCy's EntityRuler.
    
    Simulates the basic functionality of spaCy's EntityRuler pipe component,
    allowing tests to verify that patterns are properly added without needing
    to load the actual spaCy machinery.
    
    Attributes:
        patterns: List of entity patterns added to the ruler
        overwrite: Flag controlling whether existing entities should be overwritten
    """
    def __init__(self):
        self.patterns = []
        self.overwrite = False
    
    def add_patterns(self, patterns):
        """Add entity recognition patterns to the ruler."""
        self.patterns.extend(patterns)


class DummyNLP:
    """
    Mock implementation of spaCy's Language (nlp) object.
    
    Simulates the pipeline management functionality of spaCy models
    to test the integration of the EntityRecognizer with an nlp pipeline.
    
    Attributes:
        pipe_names: List of component names in the pipeline
        pipes: Dictionary mapping component names to their implementations
    """
    def __init__(self):
        self.pipe_names = []
        self.pipes = {}
    
    def add_pipe(self, component_name, *, before=None, name=None):
        """
        Mock of spaCy's add_pipe that creates and returns a ruler component.
        
        Args:
            component_name: Name of the component type to add (e.g., "entity_ruler")
            before: Name of component to insert before (keyword-only)
            name: Instance name for the added component (keyword-only)
            
        Returns:
            A DummyRuler instance representing the added pipe component
        """
        ruler = DummyRuler()
        self.pipes[name] = ruler
        self.pipe_names.append(name)
        return ruler


@pytest.fixture
def setup():
    """
    Create both EntityRecognizer and dummy NLP in one fixture.
    
    Returns:
        Dict containing instantiated test objects:
            - 'recognizer': The EntityRecognizer instance being tested
            - 'nlp': A DummyNLP mock object
    """
    return {
        "recognizer": EntityRecognizer(),
        "nlp": DummyNLP()
    }


def test_english_patterns(setup):
    """Test essential patterns for English exist."""
    recognizer = setup["recognizer"]
    patterns = recognizer.get_entity_patterns('en')
    
    assert patterns, "Should generate patterns for English"
    
    wolf_patterns = [p for p in patterns if isinstance(p["pattern"], str) and p["pattern"].lower() == "wolf"]
    assert wolf_patterns, "Should have pattern for 'wolf'"
    assert all(p["label"] == "ANIMAL_CHAR" for p in wolf_patterns), "Should label wolves as ANIMAL_CHAR"


def test_supported_languages(setup):
    """Test all supported languages generate appropriate patterns."""
    recognizer = setup["recognizer"]
    
    test_cases = [
        ('en', "wolf"),
        ('nl', "leeuw"),
        ('de', "Wolf"),
    ]
    
    for lang, animal in test_cases:
        patterns = recognizer.get_entity_patterns(lang)
        assert patterns, f"Should generate patterns for {lang}"
        
        # Check if the animal exists in at least one pattern
        found = any(
            (isinstance(p["pattern"], str) and p["pattern"].lower() == animal.lower()) or
            (isinstance(p["pattern"], list) and any(token.get("LOWER") == animal.lower() for token in p["pattern"] if isinstance(token, dict)))
            for p in patterns
        )
        assert found, f"Should include pattern for '{animal}' in {lang}"


def test_unsupported_language(setup):
    """Test behavior for unsupported languages."""
    recognizer = setup["recognizer"]
    patterns = recognizer.get_entity_patterns('xq')
    assert patterns == [], "Should return empty list for unsupported language"


def test_add_patterns_to_model(setup):
    """Test adding patterns to a spaCy model."""
    recognizer, nlp = setup["recognizer"], setup["nlp"]
    
    recognizer.add_entity_patterns(nlp, 'en')
    
    assert 'animal_ruler' in nlp.pipe_names, "Should add ruler to pipeline"
    assert nlp.pipes['animal_ruler'].patterns, "Should add patterns to ruler"
    assert nlp.pipes['animal_ruler'].overwrite, "Should set overwrite=True"


def test_add_patterns_with_existing_ruler(setup):
    """Test adding patterns when ruler already exists in pipeline."""
    recognizer, nlp = setup["recognizer"], setup["nlp"]
    
    # Add ruler first
    ruler = DummyRuler()
    nlp.pipes['animal_ruler'] = ruler
    nlp.pipe_names.append('animal_ruler')
    
    # Then try to add patterns
    recognizer.add_entity_patterns(nlp, 'en')
    
    # Should not duplicate the ruler
    assert nlp.pipe_names.count('animal_ruler') == 1, "Should not add duplicate ruler"


def test_invalid_model_handling(setup):
    """Test handling models without add_pipe method."""
    recognizer = setup["recognizer"]
    invalid_model = object()  # An object without add_pipe
    
    # Should not raise an exception
    recognizer.add_entity_patterns(invalid_model, 'en')