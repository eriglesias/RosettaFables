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
        self.moral_detector = MoralDetector(self.analysis_dir)
        self.nlp_techniques = NLPTechniques(self.analysis_dir)
        self.sentiment_analyzer = SentimentAnalyzer()
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
            self.logger.warning("No processed fables found for analysis")
            return results
        
        # Run basic analysis if any of those types are requested
        if any(analysis_type in analysis_types for analysis_type in ['pos', 'entity', 'moral', 'comparison', 'character']):
            basic_results = self._run_basic_analysis(fables_by_language, analysis_types)
            results.update(basic_results)
            
        # Clustering analysis
        if 'clustering' in analysis_types:
            clustering_results = self._run_clustering_analysis(fables_by_language)
            results['clustering'] = clustering_results

        # Sentiment analysis
        if 'sentiment' in analysis_types:
            sentiment_results = self._run_sentiment_analysis(fables_by_language)
            results['sentiment'] = sentiment_results

        # Style analysis
        if 'style' in analysis_types:
            style_results = self._run_style_analysis(fables_by_language)
            results['style'] = style_results

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
        
    def _load_processed_fables(self, languages: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load processed fables for all languages.
        
        Args:
            languages: List of language codes
            
        Returns:
            Dictionary mapping language codes to fable lists
        """
        fables_by_language = {}
        
        for lang in languages:
            processed_file = self.output_dir / "processed" / f"fables_{lang}.json"
            
            if not processed_file.exists():
                self.logger.warning("No processed data file for %s", lang)
                continue
                
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    fables = json.load(f)
                    
                    # Ensure we have a list of fables
                    if isinstance(fables, list):
                        fables_by_language[lang] = fables
                        self.logger.info("Loaded %d fables for %s", len(fables), lang)
                    else:
                        self.logger.warning("Data for %s is not in expected list format", lang)
                        
            except FileNotFoundError:
                self.logger.error("File not found: %s", processed_file)
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON in %s: %s", processed_file.name, e)
            except Exception as e:
                self.logger.error("Error loading processed data for %s: %s", lang, e)
        
        # Also prepare a fables_by_id dictionary for cross-language analysis
        fables_by_id = {}
        for lang, fables in fables_by_language.items():
            for fable in fables:
                if isinstance(fable, dict) and 'id' in fable:
                    fable_id = fable['id']
                    if fable_id not in fables_by_id:
                        fables_by_id[fable_id] = {}
                    fables_by_id[fable_id][lang] = fable
        
        self.fables_by_id = fables_by_id 
        
        return fables_by_language

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
                    
                for token_pos in fable.get('pos_tags', []):
                    if isinstance(token_pos, list) and len(token_pos) >= 2:
                        pos = token_pos[1]  # Only use the POS tag, ignore token
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
        # Use the entity_analyzer to do the work
        return self.entity_analyzer.analyze_entity_distribution(language)

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
        
        return character_stats

    def _find_common_fable_ids(self, languages: List[str]) -> List[str]:
        """
        Find fable IDs that appear in multiple languages.
        
        Args:
            languages: List of language codes
            
        Returns:
            List of fable IDs that appear in multiple languages
        """
        # We can use self.fables_by_id directly since it's already built in _load_processed_fables
        common_ids = []
        
        for fable_id, lang_fables in self.fables_by_id.items():
            if len(lang_fables) >= 2:  # Only include fables that appear in at least 2 languages
                common_ids.append(fable_id)
        
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
        # Use the fables_by_id dictionary that was created in _load_processed_fables
        if fable_id not in self.fables_by_id or len(self.fables_by_id[fable_id]) < 2:
            return None
            
        comparison = {
            'fable_id': fable_id,
            'languages': list(self.fables_by_id[fable_id].keys()),
            'title': {},
            'token_counts': {},
            'sentence_counts': {},
            'entity_counts': {},
            'pos_distribution': {},
            'has_moral': {},
            'moral_length': {},
        }
        
        # Extract comparison data for each language version
        for lang, fable in self.fables_by_id[fable_id].items():
            # Basic statistics
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
        
        return comparison

    def _run_basic_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]], analysis_types: List[str]) -> Dict[str, Any]:
        """
        Run the original basic analysis types.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            analysis_types: List of analysis types to run
            
        Returns:
            Dictionary of analysis results
        """
        results = {}
        
        # POS tag distribution analysis
        if 'pos' in analysis_types:
            pos_results = {}
            for lang in fables_by_language.keys():
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
            for lang in fables_by_language.keys():
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
            fable_ids = self._find_common_fable_ids(list(fables_by_language.keys()))
            self.logger.info("Found %d fables with content in multiple languages", len(fable_ids))
            
            for fable_id in fable_ids:
                comparison = self._compare_fable(fable_id, list(fables_by_language.keys()))
                if comparison:
                    comparison_results[fable_id] = comparison
                    self.writer.save_comparison_results(comparison, fable_id)
            
            results['fable_comparisons'] = comparison_results
        
        return results

    def _run_clustering_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run clustering analysis using the ClusteringAnalyzer.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of clustering analysis results
        """
        try:
            # Flatten all fables for clustering
            all_fables = []
            for lang, fables in fables_by_language.items():
                for fable in fables:
                    # Ensure the fable has a language attribute
                    if 'language' not in fable:
                        fable['language'] = lang
                    all_fables.append(fable)
            
            # Run K-means clustering
            kmeans_results = self.clustering_analyzer.kmeans_clustering(
                all_fables, 
                n_clusters=min(5, len(all_fables) // 2) if len(all_fables) > 5 else 2,
                feature_type='tfidf'
            )
            
            # Run hierarchical clustering
            hierarchical_results = self.clustering_analyzer.hierarchical_clustering(
                all_fables,
                feature_type='tfidf'
            )
            
            # Run DBSCAN clustering
            dbscan_results = self.clustering_analyzer.dbscan_clustering(
                all_fables,
                feature_type='tfidf'
            )
            
            # Cross-language clustering
            cross_lang_results = self.clustering_analyzer.cross_language_clustering(
                self.fables_by_id,
                feature_type='tfidf'
            )
            
            # Determine optimal number of clusters
            optimization_results = self.clustering_analyzer.optimize_clusters(
                all_fables,
                feature_type='tfidf'
            )
            
            # Combine results
            results = {
                'kmeans': kmeans_results,
                'hierarchical': hierarchical_results,
                'dbscan': dbscan_results,
                'cross_language': cross_lang_results,
                'optimal_clusters': optimization_results
            }
            
            # Save results
            self.clustering_analyzer.save_analysis('all', 'kmeans', kmeans_results)
            self.clustering_analyzer.save_analysis('all', 'hierarchical', hierarchical_results)
            self.clustering_analyzer.save_analysis('all', 'dbscan', dbscan_results)
            self.clustering_analyzer.save_analysis('all', 'cross_language', cross_lang_results)
            
            self.logger.info("Completed clustering analysis")
            return results
            
        except Exception as e:
            self.logger.error("Error in clustering analysis: %s", e)
            return {'error': str(e)}

    def _run_sentiment_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run sentiment analysis using the SentimentAnalyzer.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of sentiment analysis results
        """
        try:
            sentiment_by_language = {}
            
            # Analyze sentiment for each language
            for lang, fables in fables_by_language.items():
                lang_results = []
                
                for fable in fables:
                    sentiment_result = self.sentiment_analyzer.analyze_sentiment(fable)
                    
                    # Add fable ID for reference
                    fable_id = fable.get('id', 'unknown')
                    sentiment_result['fable_id'] = fable_id
                    
                    lang_results.append(sentiment_result)
                
                sentiment_by_language[lang] = lang_results
            
            # Compare sentiment across languages for the same fables
            sentiment_comparison = self.sentiment_analyzer.compare_sentiment_across_languages(
                self.fables_by_id
            )
            
            # Correlate sentiment with moral type
            all_fables_with_sentiment = []
            for lang_results in sentiment_by_language.values():
                all_fables_with_sentiment.extend(lang_results)
            
            sentiment_moral_correlation = self.sentiment_analyzer.correlate_sentiment_with_moral_type(
                all_fables_with_sentiment
            )
            
            # Combine results
            results = {
                'by_language': sentiment_by_language,
                'cross_language_comparison': sentiment_comparison,
                'moral_correlation': sentiment_moral_correlation
            }
            
            self.logger.info("Completed sentiment analysis")
            return results
            
        except Exception as e:
            self.logger.error("Error in sentiment analysis: %s", e)
            return {'error': str(e)}

    def _run_style_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run style analysis using the StyleAnalyzer.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of style analysis results
        """
        try:
            style_by_language = {}
            
            # Analyze style for each language
            for lang, fables in fables_by_language.items():
                lang_results = []
                
                for fable in fables:
                    # Get fable ID for reference
                    fable_id = fable.get('id', 'unknown')
                    
                    # Analyze sentence complexity
                    complexity_result = self.style_analyzer.sentence_complexity(fable)
                    
                    # Analyze lexical richness
                    richness_result = self.style_analyzer.lexical_richness(fable)
                    
                    # Analyze rhetorical devices
                    devices_result = self.style_analyzer.rhetorical_devices(fable)
                    
                    # Combine results for this fable
                    fable_result = {
                        'fable_id': fable_id,
                        'sentence_complexity': complexity_result,
                        'lexical_richness': richness_result,
                        'rhetorical_devices': devices_result
                    }
                    
                    lang_results.append(fable_result)
                
                style_by_language[lang] = lang_results
            
            self.logger.info("Completed style analysis")
            return style_by_language
            
        except Exception as e:
            self.logger.error("Error in style analysis: %s", e)
            return {'error': str(e)}

    def _run_syntax_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run syntax analysis using the SyntaxAnalyzer.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of syntax analysis results
        """
        try:
            syntax_by_language = {}
            
            # Analyze syntax for each language
            for lang, fables in fables_by_language.items():
                lang_results = []
                
                for fable in fables:
                    # Get fable ID for reference
                    fable_id = fable.get('id', 'unknown')
                    
                    # Analyze dependency frequencies
                    dep_freq_result = self.syntax_analyzer.dependency_frequencies(fable)
                    
                    # Analyze dependency distances
                    dep_dist_result = self.syntax_analyzer.dependency_distances(fable)
                    
                    # Analyze tree shapes
                    tree_result = self.syntax_analyzer.tree_shapes(fable)
                    
                    # Analyze dominant constructions
                    const_result = self.syntax_analyzer.dominant_constructions(fable)
                    
                    # Analyze semantic roles
                    roles_result = self.syntax_analyzer.semantic_roles(fable)
                    
                    # Combine results for this fable
                    fable_result = {
                        'fable_id': fable_id,
                        'dependency_frequencies': dep_freq_result,
                        'dependency_distances': dep_dist_result,
                        'tree_shapes': tree_result,
                        'dominant_constructions': const_result,
                        'semantic_roles': roles_result
                    }
                    
                    # Save analysis results
                    self.syntax_analyzer.save_analysis(fable_id, lang, 'dependency_frequencies', dep_freq_result)
                    self.syntax_analyzer.save_analysis(fable_id, lang, 'tree_shapes', tree_result)
                    
                    lang_results.append(fable_result)
                
                syntax_by_language[lang] = lang_results
            
            # Run cross-fable comparison
            syntax_comparison = self.syntax_analyzer.compare_fables(
                self.fables_by_id, 'dominant_constructions'
            )
            
            # Combine results
            results = {
                'by_language': syntax_by_language,
                'cross_fable_comparison': syntax_comparison
            }
            
            self.logger.info("Completed syntax analysis")
            return results
            
        except Exception as e:
            self.logger.error("Error in syntax analysis: %s", e)
            return {'error': str(e)}

    def _run_nlp_techniques_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run NLP techniques analysis using the NLPTechniques analyzer.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of NLP techniques analysis results
        """
        try:
            # Run TF-IDF analysis
            tfidf_results = self.nlp_techniques.tfidf_analysis(fables_by_language)
            
            # Run topic modeling
            topic_results = self.nlp_techniques.topic_modeling(
                fables_by_language, 
                n_topics=5,
                method='lda'
            )
            
            # Run word embeddings analysis
            embedding_results = self.nlp_techniques.word_embeddings(
                fables_by_language,
                model_type='word2vec'
            )
            
            # Combine results
            results = {
                'tfidf_analysis': tfidf_results,
                'topic_modeling': topic_results,
                'word_embeddings': embedding_results
            }
            
            # Save results
            self.nlp_techniques.save_analysis('all', 'tfidf', tfidf_results)
            self.nlp_techniques.save_analysis('all', 'topic_modeling', topic_results)
            
            self.logger.info("Completed NLP techniques analysis")
            return results
            
        except Exception as e:
            self.logger.error("Error in NLP techniques analysis: %s", e)
            return {'error': str(e)}

    def _run_stats_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run statistical analysis using the StatsAnalyzer.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of statistical analysis results
        """
        try:
            word_freq_by_language = {}
            
            # Analyze word frequency for each language
            for lang, fables in fables_by_language.items():
                lang_results = []
                
                for fable in fables:
                    # Get fable ID for reference
                    fable_id = fable.get('id', 'unknown')
                    
                    # Analyze word frequency
                    freq_result = self.stats_analyzer.word_frequency(fable)
                    
                    # Add fable ID
                    freq_result['fable_id'] = fable_id
                    
                    lang_results.append(freq_result)
                    
                    # Save results
                    self.stats_analyzer.save_analysis(fable_id, 'word_frequency', freq_result)
                
                word_freq_by_language[lang] = lang_results
            
            # Run chi-square test for POS distribution
            chi_square_results = self.stats_analyzer.chi_square_test(
                fables_by_language, 
                feature='pos'
            )
            
            # Compare lexical diversity across languages
            lexical_diversity = {}
            for fable_id, lang_fables in self.fables_by_id.items():
                if len(lang_fables) >= 2:  # Only compare if available in multiple languages
                    diversity_result = self.stats_analyzer.compare_lexical_diversity(lang_fables)
                    lexical_diversity[fable_id] = diversity_result
            
            # Combine results
            results = {
                'word_frequency': word_freq_by_language,
                'chi_square_test': chi_square_results,
                'lexical_diversity': lexical_diversity
            }
            
            self.logger.info("Completed statistical analysis")
            return results
            
        except Exception as e:
            self.logger.error("Error in statistical analysis: %s", e)
            return {'error': str(e)}

    def _run_cross_language_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run cross-language analysis comparing the same fable across languages.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of cross-language analysis results
        """
        try:
            # Use the MoralDetector to compare morals across languages
            moral_comparison = self.moral_detector.compare_morals(self.fables_by_id)
            
            # Use the entity analyzer to compare entity distributions
            entity_results = {}
            for lang in fables_by_language.keys():
                entity_dist = self.entity_analyzer.analyze_entity_distribution(lang)
                entity_results[lang] = entity_dist
            
            # Compare word usage across languages
            word_usage_comparison = {}
            for fable_id, lang_fables in self.fables_by_id.items():
                if len(lang_fables) >= 2:
                    usage_result = self.stats_analyzer.compare_word_usage(lang_fables)
                    word_usage_comparison[fable_id] = usage_result
            
            # Combine results
            results = {
                'moral_comparison': moral_comparison,
                'entity_distribution': entity_results,
                'word_usage_comparison': word_usage_comparison
            }
            
            self.logger.info("Completed cross-language analysis")
            return results
            
        except Exception as e:
            self.logger.error("Error in cross-language analysis: %s", e)
            return {'error': str(e)}