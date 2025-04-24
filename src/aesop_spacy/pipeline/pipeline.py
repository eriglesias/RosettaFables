# src/aesop_spacy/pipeline/pipeline.py
"""
Main pipeline module for processing and analyzing Aesop's fables.

This module orchestrates the complete fable processing pipeline including:
- Loading fables from files
- Cleaning and normalizing text
- Extracting relevant content
- Processing with NLP models
- Recognizing custom entities
- Analyzing linguistic features
- Generating output files

The pipeline supports multiple languages and provides comparison
capabilities across translations of the same fable.
"""

from typing import Dict, List, Any
from pathlib import Path
import logging
import json
from aesop_spacy.io.loader import FableLoader
from aesop_spacy.io.writer import OutputWriter
from aesop_spacy.io.serializer import SpacySerializer
from aesop_spacy.preprocessing.cleaner import TextCleaner
from aesop_spacy.preprocessing.extractor import ContentExtractor
from aesop_spacy.preprocessing.processor import FableProcessor
from aesop_spacy.models.model_manager import get_model
from aesop_spacy.preprocessing.entity_recognizer import EntityRecognizer
# ----- imports from analysis folder
from aesop_spacy.analysis.clustering import ClusteringAnalyzer
from aesop_spacy.analysis.entity_analyzer import EntityAnalyzer
from aesop_spacy.analysis.moral_detector import MoralDetector
from aesop_spacy.analysis.nlp_techniques import NLPTechniques
from aesop_spacy.analysis.sentiment_analyzer import SentimentAnalyzer
from aesop_spacy.analysis.stats_analyzer import StatsAnalyzer
from aesop_spacy.analysis.style_analyzer import StyleAnalyzer
from aesop_spacy.analysis.syntax_analyzer import SyntaxAnalyzer
class FablePipeline:
    """Coordinates the entire fable processing pipeline."""

    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize the pipeline with directories and components.
        
        Args:
            data_dir: Root data directory
            output_dir: Directory for output files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.analysis_dir = output_dir / "analysis"
        self.logger = logging.getLogger(__name__)

        # Configure logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Initialize components
        self.loader = FableLoader(data_dir)
        self.cleaner = TextCleaner()
        self.extractor = ContentExtractor()
        self.processor = FableProcessor()
        self.serializer = SpacySerializer()
        self.recognizer = EntityRecognizer()
        self.writer = OutputWriter(output_dir)
        self.clustering_analyzer = ClusteringAnalyzer(self.analysis_dir)
        self.entity_analyzer = EntityAnalyzer(self.analysis_dir)
        self.moral_dector = MoralDetector(self.analysis_dir)
        self.nlp_techniques = NLPTechniques(self.analysis_dir)
        self.sentiment_analyzer = SentimentAnalyzer() #needing analysis_dir?
        self.stats_analyzer = StatsAnalyzer(self.analysis_dir)
        self.style_analyzer = StyleAnalyzer(self.analysis_dir)
        self.syntax_analyzer = SyntaxAnalyzer(self.analysis_dir)
        
        # printing logging 
        self.logger.info("Fable pipeline initialized")
        self.logger.info("Data directory: %s", data_dir)
        self.logger.info("Output directory: %s", output_dir)

    def run(self, use_processed=True):
        """
        Run the complete pipeline from loading to processing to analysis.
        
        Args:
            use_processed: If True, load from processed files when available.
            If False, always process from raw files.
        
        Returns:
            True if the pipeline completed successfully
        """
        self.logger.info("Starting fable processing pipeline (use_processed=%s)", use_processed)

        # Check if ALL expected language files exist
        all_files_exist = use_processed and self._processed_files_exist(check_all_languages=True)
        
        if all_files_exist:
            # All files exist, just load them
            fables_by_language = self._load_from_processed()
            # Log what we found
            total_fables = sum(len(fables) for fables in fables_by_language.values())
            self.logger.info("Loaded %d fables from processed files across %d languages", 
                            total_fables, len(fables_by_language))
            return True
        
        # Regular processing with the loader which will handle missing files
        self.logger.info("Processing from raw files")
        fables_by_language = self.loader.load_all()
        
        # Log what we found
        total_fables = sum(len(fables) for fables in fables_by_language.values())
        self.logger.info("Loaded %d fables across %d languages", 
                        total_fables, len(fables_by_language))

        # Process each language
        for lang, fables in fables_by_language.items():
            self._process_language(lang, fables)

        self.logger.info("Pipeline execution completed successfully")
        return True

    def _processed_files_exist(self, check_all_languages=True):
        """
        Check if processed files already exist.
        
        Args:
            check_all_languages: If True, check that ALL expected language files exist
                                If False, check if ANY language files exist
        """
        processed_dir = self.output_dir / "processed"
        if not processed_dir.exists():
            return False
            
        # Expected languages
        expected_langs = ['en', 'de', 'nl', 'es', 'grc']
        
        if check_all_languages:
            # Check that ALL expected languages exist
            for lang in expected_langs:
                if not (processed_dir / f"fables_{lang}.json").exists():
                    self.logger.info("Missing language file: %s", lang)
                    return False
            return True
        else:
            # Check if ANY language files exist
            json_files = list(processed_dir.glob("fables_*.json"))
            return len(json_files) > 0

    def _load_from_processed(self):
        """Load fables directly from processed JSON files."""
        fables_by_language = {}
        processed_dir = self.output_dir / "processed"

        for json_file in processed_dir.glob("fables_*.json"):
            lang = json_file.stem.split('_')[1]  # Extract language from filename

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    fables = json.load(f)

                if isinstance(fables, list):
                    fables_by_language[lang] = fables
                    self.logger.info("Loaded %d processed fables for %s", len(fables), lang)
                else:
                    self.logger.warning("Skipping %s - data not in expected list format", json_file)

            except FileNotFoundError:
                self.logger.error("File not found: %s", json_file)
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON in %s: %s", json_file.name, e)
            except IOError as e:
                self.logger.error("I/O error reading %s: %s", json_file.name, e)

        return fables_by_language

    def _process_language(self, language: str, fables: List[Dict[str, Any]]):
        """
        Process all fables for a specific language.
        
        Args:
            language: Language code
            fables: List of fables for this language
        """
        if not fables:
            self.logger.warning("No fables to process for language: %s", language)
            return

        self.logger.info("Processing %d fables for language: %s", len(fables), language)

        # Get the appropriate model
        model = get_model(language)
        if not model:
            self.logger.warning("Skipping %s - no NLP model available", language)
            return

        # Get model info differently depending on model type
        if hasattr(model, 'meta'):
            # This is a spaCy model
            model_name = model.meta.get('name', 'unknown')
            
            # First process a sample fable to get canonical forms
            if fables and (not hasattr(self.cleaner, 'canonical_forms') or not self.cleaner.canonical_forms):
                # Process first fable to get canonical forms
                sample_fable = fables[0]
                cleaned_sample = self.cleaner.clean_fable(sample_fable)
                # Now cleaner should have canonical_forms available
            
            # Get canonical forms from cleaner
            canonical_forms = {}
            if hasattr(self.cleaner, 'canonical_forms'):
                canonical_forms = self.cleaner.canonical_forms
                
            # Add entity patterns to enhance the model
            self.recognizer.add_entity_patterns(model, language, canonical_forms)
            
            # Add character consolidation component to normalize entity mentions
            self.recognizer.add_character_consolidation(model)
            
            self.logger.info("Enhanced model %s with custom entity recognition", model_name)
        elif hasattr(model, 'nlp'):
            # This is likely a StanzaWrapper
            model_name = f"StanzaWrapper({language})"
        else:
            # Fallback for any other model type
            model_name = f"{type(model).__name__}"

        self.logger.info("Using model %s for %s", model_name, language)

        # Process each fable
        processed_fables = []
        for i, fable in enumerate(fables):
            try:
                # Clean the text
                cleaned_fable = self.cleaner.clean_fable(fable)
                self.logger.debug("Cleaned fable: %s", cleaned_fable.get('title', 'Untitled'))

                # Extract content
                extracted_fable = self.extractor.extract_content(cleaned_fable)
                self.logger.debug("Extracted content for: %s", extracted_fable.get('title', 'Untitled'))

                # Process with NLP
                processed_fable = self.processor.process_fable(extracted_fable, model)
                self.logger.debug("Processed fable: %s", processed_fable.get('title', 'Untitled'))
                
                # Track entities if they exist (for statistics)
                document_id = fable.get('id', f"{language}_{i}")
                if 'entities' in processed_fable:
                    for entity_data in processed_fable['entities']:
                        if len(entity_data) >= 2:  # Ensure we have at least [text, label]
                            entity_text, entity_label = entity_data[0], entity_data[1]
                            self.recognizer.track_entity(entity_text, entity_label, document_id)

                # Serialize spaCy objects to JSON-compatible format
                serialized_fable = self.serializer.serialize(processed_fable)
                processed_fables.append(serialized_fable)

            except ValueError as e:
                self.logger.error("Value error processing fable %s: %s", 
                                 fable.get('title', 'Untitled'), e)
            except KeyError as e:
                self.logger.error("Missing key in fable %s: %s", 
                                 fable.get('title', 'Untitled'), e)
            except RuntimeError as e:
                self.logger.error("Runtime error processing fable %s: %s", 
                                 fable.get('title', 'Untitled'), e)
            except Exception as e:
                self.logger.error("Unexpected error processing fable %s: %s (%s)", 
                                 fable.get('title', 'Untitled'), e, type(e).__name__)

        # Save results
        if processed_fables:
            output_file = self.writer.save_processed_fables(processed_fables, language)
            self.logger.info("Saved %d processed fables to %s", len(processed_fables), output_file)
            
        # Save entity statistics 
        entity_stats = self.recognizer.get_entity_statistics()
        if entity_stats:
            self.writer.save_analysis_results(entity_stats, language, 'entity_stats')
            self.logger.info("Saved entity statistics for %s", language)

    def analyze(self, analysis_types=None):
        """
        Run analyses on processed fables.
        
        Args:
            analysis_types: List of analysis types to run, or None for all
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Starting fable analysis")

        all_analysis_types = [
            
            'pos', 'entity', 'moral', 'comparison', 'character', 'clustering', 'sentiment',
            'style', 'syntax', 'nlp_techniques', 'stats', 'cross_language'
        ]

        # Default to all analysis types if none specified
        if analysis_types is None:
            analysis_types = all_analysis_types       
        

        # Results container
        results = {}
        

        # Load processed fables for each language
        languages = []
        for lang_file in (self.output_dir / "processed").glob("fables_*.json"):
            lang = lang_file.stem.split('_')[1]
            languages.append(lang)  
        self.logger.info("Found processed data for languages: %s", ', '.join(languages))
        
        
        # Loading fables by language for analysis
        fables_by_language = self._load_processed_fables(languages)
        
        # Skip if no data
        if not fables_by_language:
            self.logger.warning("No processed fables found for anaylsis")
            return results
        
        if any(analysis_type in analysis_types for analysis_type in ['pos', 'entity', 'moral', 'comparison', 'charachter']):
            basic_results = self._run_basic_analysis(fables_by_language, analysis_types)
            results.update(basic_results)
        # Running the analysis from directory

        # Clustering analysis
        if 'clustering' in analysis_types:
            clustering_results = self._run_clustering_analysis(fables_by_language)
            results['clustering'] = clustering_results

        # Sentiment analysis
        if 'sentiment' in analysis_types:
            sentiment_results = self.run_sentiment_analysis(fables_by_language)
            results['sentiment'] = sentiment_results

          # Syntax analysis
        if 'syntax' in analysis_types:
            syntax_results = self._run_syntax_analysis(fables_by_language)
            results['syntax'] = syntax_results
        
        # NLP techniques analysis
        if 'nlp_techniques' in analysis_types:
            nlp_results = self._run_nlp_techniques_analysis(fables_by_language)
            results['nlp_techniques'] = nlp_results
        
        # Statistical analysis
        if 'stats' in analysis_types:
            stats_results = self._run_stats_analysis(fables_by_language)
            results['stats'] = stats_results
        
        # Cross-language analysis
        if 'cross_language' in analysis_types:
            cross_lang_results = self._run_cross_language_analysis(fables_by_language)
            results['cross_language'] = cross_lang_results
        
        self.logger.info("Analysis completed successfully")
        return results

        
        # POS tag distribution analysis
        if 'pos' in analysis_types:
            pos_results = {}
            for lang in languages:
                pos_dist = self._analyze_pos_distribution(lang)
                pos_results[lang] = pos_dist
            
            results['pos_distribution'] = pos_results
            
            # Save analysis results
            for lang, dist in pos_results.items():
                if dist:
                    self.writer.save_analysis_results(dist, lang, 'pos')
        
        # Entity analysis
        if 'entity' in analysis_types:
            entity_results = {}
            for lang in languages:
                entity_dist = self._analyze_entity_distribution(lang)
                entity_results[lang] = entity_dist
            
            results['entity_distribution'] = entity_results
            
            # Save analysis results
            for lang, dist in entity_results.items():
                if dist:
                    self.writer.save_analysis_results(dist, lang, 'entity')
        
        # Character analysis using the tracked entities from EntityRecognizer
        if 'character' in analysis_types:
            character_results = self._analyze_character_distribution()
            if character_results:
                results['character_distribution'] = character_results
                self.writer.save_analysis_results(character_results, 'all', 'character')
        
        # Cross-language fable comparison
        if 'comparison' in analysis_types:
            comparison_results = {}
            
            # Identify fable IDs that appear in multiple languages
            fable_ids = self._find_common_fable_ids(languages)
            self.logger.info("Found %d fables with content in multiple languages", len(fable_ids))
            
            for fable_id in fable_ids:
                comparison = self._compare_fable(fable_id, languages)
                if comparison:
                    comparison_results[fable_id] = comparison
                    self.writer.save_comparison_results(comparison, fable_id)
            
            results['fable_comparisons'] = comparison_results
        
        self.logger.info("Analysis completed successfully")
        return results
    
    def _analyze_character_distribution(self) -> Dict[str, Any]:
        """
        Analyze character distribution using the EntityRecognizer's tracked entities.
        
        Returns:
            Dictionary with character statistics
        """
        # Get entity statistics from the recognizer
        entity_stats = self.recognizer.get_entity_statistics()
        
        # If no stats, return empty dict
        if not entity_stats:
            return {}
            
        # Extract character information
        character_stats = {}
        
        # Look for animal characters (label "ANIMAL_CHAR")
        if "ANIMAL_CHAR" in entity_stats:
            animal_chars = entity_stats["ANIMAL_CHAR"]
            
            # Sort characters by mention count
            sorted_chars = sorted(
                animal_chars.items(),
                key=lambda x: x[1]["mentions"],
                reverse=True
            )
            
            character_stats["animals"] = {
                char: {
                    "mentions": data["mentions"],
                    "documents": data["document_count"]
                }
                for char, data in sorted_chars
            }
            
            # Calculate co-occurrence statistics (for future implementation)
            # This would require tracking co-occurrences in EntityRecognizer
            
        return character_stats
    
    def _analyze_pos_distribution(self, language: str) -> Dict[str, float]:
        """
        Analyze part-of-speech distribution for a language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with POS tag frequencies as percentages
        """
        try:
            # Load processed fables
            processed_file = self.output_dir / "processed" / f"fables_{language}.json"
            
            if not processed_file.exists():
                self.logger.warning("No processed data for %s", language)
                return {}
                
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Ensure data is in the expected format
                if not isinstance(data, list):
                    self.logger.warning("Data for %s is not in expected list format", language)
                    return {}
                
                fables = data
                
            # Count POS tags
            pos_counts = {}
            total_tokens = 0
            
            for fable in fables:
                if not isinstance(fable, dict):
                    continue
                    
                for _, pos in fable.get('pos_tags', []):
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    total_tokens += 1
            
            # Convert to percentages
            if total_tokens > 0:
                pos_distribution = {pos: (count / total_tokens * 100) 
                                   for pos, count in pos_counts.items()}
                
                # Sort by frequency (descending)
                pos_distribution = dict(sorted(pos_distribution.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True))
                
                self.logger.info("Analyzed POS distribution for %s: %d tags from %d tokens", 
                                language, len(pos_distribution), total_tokens)
                return pos_distribution
            else:
                self.logger.warning("No tokens found for %s", language)
                return {}
                
        except FileNotFoundError:
            self.logger.error("File not found for %s", language)
            return {}
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON for %s: %s", language, e)
            return {}
        except Exception as e:
            self.logger.error("Error analyzing POS distribution for %s: %s (%s)", 
                             language, e, type(e).__name__)
            return {}
    
    def _analyze_entity_distribution(self, language: str) -> Dict[str, Dict[str, float]]:
        """
        Analyze named entity distribution for a language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with entity type frequencies and examples
        """
        try:
            # Load processed fables
            processed_file = self.output_dir / "processed" / f"fables_{language}.json"
            
            if not processed_file.exists():
                self.logger.warning("No processed data for %s", language)
                return {}
                
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Ensure data is in the expected format
                if not isinstance(data, list):
                    self.logger.warning("Data for %s is not in expected list format", language)
                    return {}
                
                fables = data
                
            # Count entity types
            entity_counts = {}
            entity_examples = {}
            total_entities = 0
            
            for fable in fables:
                if not isinstance(fable, dict):
                    continue
                    
                for entity_data in fable.get('entities', []):
                    # Make sure entity data has at least two elements
                    if not isinstance(entity_data, list) or len(entity_data) < 2:
                        continue
                        
                    entity, entity_type = entity_data[0], entity_data[1]
                    
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    
                    # Store examples for each entity type (up to 5)
                    if entity_type not in entity_examples:
                        entity_examples[entity_type] = []
                    
                    if len(entity_examples[entity_type]) < 5 and entity not in entity_examples[entity_type]:
                        entity_examples[entity_type].append(entity)
                        
                    total_entities += 1
            
            # Combine counts and examples
            entity_distribution = {}
            
            if total_entities > 0:
                for entity_type, count in entity_counts.items():
                    entity_distribution[entity_type] = {
                        'percentage': count / total_entities * 100,
                        'count': count,
                        'examples': entity_examples.get(entity_type, [])
                    }
                
                # Sort by frequency (descending)
                entity_distribution = dict(sorted(entity_distribution.items(), 
                                                key=lambda x: x[1]['count'], 
                                                reverse=True))
                
                self.logger.info("Analyzed entity distribution for %s: %d types from %d entities", 
                                language, len(entity_distribution), total_entities)
                return entity_distribution
            else:
                self.logger.warning("No entities found for %s", language)
                return {}
                
        except FileNotFoundError:
            self.logger.error("File not found for %s", language)
            return {}
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON for %s: %s", language, e)
            return {}
        except Exception as e:
            self.logger.error("Error analyzing entity distribution for %s: %s (%s)", 
                             language, e, type(e).__name__)
            return {}
    
    def _find_common_fable_ids(self, languages: List[str]) -> List[str]:
        """
        Find fable IDs that appear in multiple languages.
        
        Args:
            languages: List of language codes
            
        Returns:
            List of fable IDs that appear in multiple languages
        """
        # Load all fable data
        fables_by_language = {}
        
        for lang in languages:
            processed_file = self.output_dir / "processed" / f"fables_{lang}.json"
            
            if not processed_file.exists():
                continue
                
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check if the loaded data is a list (as expected)
                    if isinstance(data, list):
                        fables_by_language[lang] = data
                    else:
                        self.logger.warning("Skipping %s - data not in expected list format", lang)
            except FileNotFoundError:
                self.logger.error("File not found: %s", processed_file)
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON in %s: %s", processed_file.name, e)
            except Exception as e:
                self.logger.error("Error loading processed data for %s: %s (%s)", 
                                 lang, e, type(e).__name__)
        
        # Find IDs that appear in multiple languages
        fable_ids_by_language = {}
        
        for lang, fables in fables_by_language.items():
            # Make sure each fable is a dictionary with an 'id' key
            fable_ids = []
            for fable in fables:
                if isinstance(fable, dict) and fable.get('id'):
                    fable_ids.append(fable.get('id'))
            
            fable_ids_by_language[lang] = set(fable_ids)
        
        # Find IDs that appear in at least 2 languages
        if not fable_ids_by_language:
            return []
            
        all_ids = set().union(*fable_ids_by_language.values()) if fable_ids_by_language else set()
        common_ids = [fable_id for fable_id in all_ids 
                     if sum(1 for lang_ids in fable_ids_by_language.values() 
                           if fable_id in lang_ids) >= 2]
        
        return common_ids
    
    def _compare_fable(self, fable_id: str, languages: List[str]) -> Dict[str, Any]:
        """
        Compare the same fable across different languages.
        
        Args:
            fable_id: Fable ID to compare
            languages: List of language codes to check
            
        Returns:
            Comparison data dictionary or None if not found in multiple languages
        """
        comparison = {
            'fable_id': fable_id,
            'languages': [],
            'title': {},
            'token_counts': {},
            'sentence_counts': {},
            'entity_counts': {},
            'pos_distribution': {},
            'has_moral': {},
            'moral_length': {},
        }
        
        # Load processed data for each language
        for lang in languages:
            processed_file = self.output_dir / "processed" / f"fables_{lang}.json"
            
            if not processed_file.exists():
                continue
                
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Ensure data is in the expected format
                    if not isinstance(data, list):
                        self.logger.warning("Data for %s is not in expected list format - skipping comparison", lang)
                        continue
                    
                    fables = data
                    
                # Find the specific fable
                for fable in fables:
                    if not isinstance(fable, dict):
                        continue
                        
                    if fable.get('id') == fable_id:
                        # Add language to the list
                        comparison['languages'].append(lang)
                        
                        # Extract comparison data
                        comparison['title'][lang] = fable.get('title', '')
                        comparison['token_counts'][lang] = len(fable.get('tokens', []))
                        comparison['sentence_counts'][lang] = len(fable.get('sentences', []))
                        comparison['entity_counts'][lang] = len(fable.get('entities', []))
                        
                        # Calculate POS distribution
                        pos_counts = {}
                        for token_pos in fable.get('pos_tags', []):
                            if isinstance(token_pos, list) and len(token_pos) >= 2:
                                pos = token_pos[1]  # Only use the POS tag, ignore token
                                pos_counts[pos] = pos_counts.get(pos, 0) + 1
                            
                        total_tokens = sum(pos_counts.values())
                        if total_tokens > 0:
                            comparison['pos_distribution'][lang] = {
                                pos: count / total_tokens * 100 
                                for pos, count in pos_counts.items()
                            }
                        
                        # Check moral
                        moral = fable.get('moral', {})
                        if isinstance(moral, dict):
                            has_moral = bool(moral.get('text', ''))
                            comparison['has_moral'][lang] = has_moral
                            
                            if has_moral:
                                moral_text = moral.get('text', '')
                                comparison['moral_length'][lang] = len(moral_text.split())
                        
                        break
                    
            except FileNotFoundError:
                self.logger.error("File not found: %s", processed_file)
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON in %s: %s", processed_file.name, e)
            except Exception as e:
                self.logger.error("Error comparing fable %s for %s: %s (%s)", 
                                 fable_id, lang, e, type(e).__name__)
        
        # Only return if we found the fable in at least 2 languages
        if len(comparison['languages']) >= 2:
            self.logger.info("Compared fable %s across %d languages", 
                            fable_id, len(comparison['languages']))
            return comparison
        else:
            self.logger.warning("Fable %s not found in multiple languages", fable_id)
            return None