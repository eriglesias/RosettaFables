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

from typing import Dict, List, Any, Set
from pathlib import Path
import logging
import json
from aesop_spacy.io.loader import FableLoader
from aesop_spacy.io.writer import OutputWriter
from aesop_spacy.io.serializer import SpacySerializer
from aesop_spacy.preprocessing.cleaner import TextCleaner
from aesop_spacy.preprocessing.extractor import ContentExtractor
from aesop_spacy.preprocessing.processor import FableProcessor
from aesop_spacy.models.model_manager import get_model, verify_models
from aesop_spacy.preprocessing.entity_recognizer import EntityRecognizer
from aesop_spacy.utils.log_utils import section_header, subsection_header, log_timing, wrap_analysis_result, format_count


class FablePipeline:
    """Coordinates the entire fable processing pipeline with lazy loading of components."""

    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize the pipeline with directories and core components.
        
        Args:
            data_dir: Root data directory
            output_dir: Directory for output files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.analysis_dir = output_dir / "analysis"
        self.logger = logging.getLogger(__name__)

        # Initializing empty dictionaries
        self.fables_by_language = {}
        self.fables_by_id = {}

        # Configure logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Initialize core components that are always needed
        self.loader = FableLoader(data_dir)
        self.cleaner = TextCleaner()
        self.extractor = ContentExtractor()
        self.processor = FableProcessor()
        self.serializer = SpacySerializer()
        self.recognizer = EntityRecognizer()
        self.writer = OutputWriter(output_dir)

        # Initialize transformer manager early if needed for multiple components
        self._transformer_manager = None
        # Lazy-loaded components - will be initialized on first use
        self._clustering_analyzer = None
        self._entity_analyzer = None
        self._moral_detector = None
        self._nlp_techniques = None
        self._sentiment_analyzer = None
        self._stats_analyzer = None
        self._style_analyzer = None
        self._syntax_analyzer = None
        self._pos_analyzer = None
        self._comparison_analyzer = None

        # Print logging info
        self.logger.info("Fable pipeline initialized")
        self.logger.info("Data directory: %s", data_dir)
        self.logger.info("Output directory: %s", output_dir)

    # Lazy-loaded property getters
    @property
    def transformer_manager(self):
        """Lazily load the transformer manager"""
        if self._transformer_manager is None:
            from aesop_spacy.models.transformer_manager import TransformerManager
            self.logger.debug("Initializing TransformerManager")
            self._transformer_manager = TransformerManager()
        return self._transformer_manager


    @property
    def clustering_analyzer(self):
        """Lazily load the clustering analyzer"""
        if self._clustering_analyzer is None:
            from aesop_spacy.analysis.clustering import ClusteringAnalyzer
            self.logger.debug("Initializing ClusteringAnalyzer")
            self._clustering_analyzer = ClusteringAnalyzer(self.analysis_dir)
        return self._clustering_analyzer


    @property
    def entity_analyzer(self):
        """Lazily load the entity analyzer"""
        if self._entity_analyzer is None:
            from aesop_spacy.analysis.entity_analyzer import EntityAnalyzer
            self.logger.debug("Initializing EntityAnalyzer")
            self._entity_analyzer = EntityAnalyzer(self.analysis_dir)
        return self._entity_analyzer
    

    @property
    def moral_detector(self):
        """Lazily load the moral detector"""
        if self._moral_detector is None:
            from aesop_spacy.analysis.moral_detector import MoralDetector
            self.logger.debug("Initializing MoralDetector")
            self._moral_detector = MoralDetector(self.analysis_dir)
        return self._moral_detector
    

    @property
    def nlp_techniques(self):
        """Lazily load the NLP techniques analyzer"""
        if self._nlp_techniques is None:
            from aesop_spacy.analysis.nlp_techniques import NLPTechniques
            self.logger.debug("Initializing NLPTechniques")
            self._nlp_techniques = NLPTechniques(self.analysis_dir)
        return self._nlp_techniques


    @property
    def sentiment_analyzer(self):
        """Lazily load the sentiment analyzer"""
        if self._sentiment_analyzer is None:
            from aesop_spacy.analysis.sentiment_analyzer import SentimentAnalyzer
            self.logger.debug("Initializing SentimentAnalyzer")
            self._sentiment_analyzer = SentimentAnalyzer(transformer_manager=self.transformer_manager)
        return self._sentiment_analyzer
    

    @property
    def stats_analyzer(self):
        """Lazily load the stats analyzer"""
        if self._stats_analyzer is None:
            from aesop_spacy.analysis.stats_analyzer import StatsAnalyzer
            self.logger.debug("Initializing StatsAnalyzer")
            self._stats_analyzer = StatsAnalyzer(self.analysis_dir)
        return self._stats_analyzer
    

    @property
    def style_analyzer(self):
        """Lazily load the style analyzer"""
        if self._style_analyzer is None:
            from aesop_spacy.analysis.style_analyzer import StyleAnalyzer
            self.logger.debug("Initializing StyleAnalyzer")
            self._style_analyzer = StyleAnalyzer(self.analysis_dir)
        return self._style_analyzer
    

    @property
    def syntax_analyzer(self):
        """Lazily load the syntax analyzer"""
        if self._syntax_analyzer is None:
            from aesop_spacy.analysis.syntax_analyzer import SyntaxAnalyzer
            self.logger.debug("Initializing SyntaxAnalyzer")
            self._syntax_analyzer = SyntaxAnalyzer(self.analysis_dir)
        return self._syntax_analyzer
    

    @property
    def pos_analyzer(self):
        """Lazily load the POS analyzer"""
        if self._pos_analyzer is None:
            from aesop_spacy.analysis.pos_analyzer import POSAnalyzer
            self.logger.debug("Initializing POSAnalyzer")
            self._pos_analyzer = POSAnalyzer(self.analysis_dir)
        return self._pos_analyzer


    @property
    def comparison_analyzer(self):
        """Lazily load the comparison analyzer"""
        if self._comparison_analyzer is None:
            from aesop_spacy.analysis.comparison_analyzer import ComparisonAnalyzer
            self.logger.debug("Initializing ComparisonAnalyzer")
            self._comparison_analyzer = ComparisonAnalyzer(self.analysis_dir)
        return self._comparison_analyzer

    def _detect_languages(self) -> Set[str]:
        """
        Detect language codes from available fable files.
        
        Returns:
            Set of detected language codes
        """
        languages = set()
        
        # Check the fables directory for .md files
        fable_dir = self.data_dir / "fables"
        if not fable_dir.exists():
            self.logger.warning("Fables directory not found: %s", fable_dir)
            return languages
            
        # Look for language tags in MD files
        for file_path in fable_dir.glob('*.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract all language tags
                    import re
                    lang_tags = re.findall(r'<language>(.*?)</language>', content)
                    for lang in lang_tags:
                        languages.add(lang.strip())
                        
            except Exception as e:
                self.logger.error("Error extracting languages from %s: %s", file_path, e)
        
        # If no languages detected, use common defaults
        if not languages:
            self.logger.warning("No languages detected in files, using defaults")
            languages = {'en', 'de', 'nl', 'es', 'grc'}
            
        return languages

    @log_timing
    def run(self, use_processed=True, verify_models=True):
        """
        Run the complete pipeline from loading to processing to analysis.
        
        Args:
            use_processed: If True, load from processed files when available.
            verify_models: If True, verify required models before processing
            
        Returns:
            True if the pipeline completed successfully
        """
        self.logger.info(section_header("FABLE PROCESSING PIPELINE"))
        self.logger.info("Starting pipeline (use_processed=%s, verify_models=%s)", 
                        use_processed, verify_models)

        # Verify models if requested
        if verify_models and not use_processed:
            self.logger.info(subsection_header("VERIFYING LANGUAGE MODELS"))
            
            # Detect languages from fable files
            languages = self._detect_languages()
            self.logger.info("Detected languages: %s", ", ".join(languages))
            
            # Verify models for detected languages
            verification = verify_models(languages=list(languages))
            
            # Handle missing models
            if verification['missing']:
                self.logger.warning("Missing models for languages: %s", 
                                  ", ".join(verification['missing']))
                self.logger.warning("Please install the missing models with:")
                for cmd in verification['install_commands']:
                    self.logger.warning("  %s", cmd)
                
                # Strict verification - don't continue if models are missing
                if not use_processed:
                    self.logger.error("Cannot process raw files without required models")
                    return False
            else:
                self.logger.info("All required language models are installed")

        # Check if ALL expected language files exist
        all_files_exist = use_processed and self._processed_files_exist(check_all_languages=True)

        if all_files_exist:
            # All files exist, just load them
            self.logger.info(subsection_header("LOADING PROCESSED FILES"))
            fables_by_language = self._load_from_processed()
            # Log what we found
            total_fables = sum(len(fables) for fables in fables_by_language.values())
            self.logger.info("Loaded %s from %s", 
                            format_count("fable", total_fables), 
                            format_count("language", len(fables_by_language)))
            return True

        # Regular processing with the loader which will handle missing files
        self.logger.info(subsection_header("PROCESSING RAW FILES"))
        fables_by_language = self.loader.load_all()

        # Log what we found
        total_fables = sum(len(fables) for fables in fables_by_language.values())
        self.logger.info("Loaded %s across %s", 
                        format_count("fable", total_fables), 
                        format_count("language", len(fables_by_language)))

        # Process each language
        for lang, fables in fables_by_language.items():
            self.logger.info(subsection_header(f"PROCESSING {lang.upper()} FABLES"))
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
            except Exception as e:
                self.logger.error("Unexpected error loading %s: %s (%s)", 
                                 json_file.name, e, type(e).__name__)

        return fables_by_language

    @log_timing
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

            # Initialize canonical forms if needed
            if fables and (not hasattr(self.cleaner, 'canonical_forms') or not self.cleaner.canonical_forms):
                # Process first fable to populate canonical forms in the cleaner
                sample_fable = fables[0]
                self.cleaner.clean_fable(sample_fable)  # Result is used to update self.cleaner.canonical_forms

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

                # Check if dependencies were extracted
                if 'sentences' in processed_fable:
                    total_deps = sum(len(s.get('dependencies', [])) for s in processed_fable['sentences'])
                    if total_deps == 0 and 'dependencies' not in processed_fable:
                        self.logger.warning("No dependencies found in processed fable %s. Check model configuration.", 
                                          fable.get('id', i))

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
            try:
                output_file = self.writer.save_processed_fables(processed_fables, language)
                self.logger.info("Saved %d processed fables to %s", len(processed_fables), output_file)
            except IOError as e:
                self.logger.error("Failed to save processed fables: %s", e)
          
        # Save entity statistics 
        entity_stats = self.recognizer.get_entity_statistics()
        if entity_stats:
            try:
                self.writer.save_analysis_results(entity_stats, language, 'entity_stats')
                self.logger.info("Saved entity statistics for %s", language)
            except IOError as e:
                self.logger.error("Failed to save entity statistics: %s", e)

    @log_timing
    def analyze(self, analysis_types=None):
        """
        Run analyses on processed fables with improved logging.
        
        Args:
            analysis_types: List of analysis types to run, or None for all
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info(section_header("FABLE ANALYSIS"))

        all_analysis_types = [
            'pos', 'entity', 'moral', 'comparison', 'clustering', 'sentiment',
            'style', 'syntax', 'nlp_techniques', 'stats', 'cross_language'
        ]

        # Default to all analysis types if none specified
        if analysis_types is None:
            analysis_types = all_analysis_types
            self.logger.info("Running all %d analysis types", len(all_analysis_types))
        else:
            self.logger.info("Running %d specified analysis types: %s", len(analysis_types), ', '.join(analysis_types))
          
        # Results container
        results = {}
      
        # Load processed fables for each language
        self.logger.info(subsection_header("LOADING DATA"))
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
       
        # Run analyses based on requested types
        if any(analysis_type in analysis_types for analysis_type in ['pos', 'entity', 'moral', 'comparison']):
            self.logger.info(subsection_header("BASIC LINGUISTIC ANALYSIS"))
            try:
                basic_results = self._run_basic_analysis(fables_by_language, analysis_types)
                results.update(basic_results)
            except Exception as e:
                self.logger.error("Error in basic linguistic analysis: %s", e)
                results['basic_analysis_error'] = str(e)
           
        # Clustering analysis
        if 'clustering' in analysis_types:
            self.logger.info(subsection_header("CLUSTERING ANALYSIS"))
            self.logger.info("Grouping similar fables across languages...")
            clustering_results = self._run_clustering_analysis(fables_by_language)
            results['clustering'] = wrap_analysis_result(
                clustering_results, "Clustering", self.logger)

        # Sentiment analysis
        if 'sentiment' in analysis_types:
            self.logger.info(subsection_header("SENTIMENT ANALYSIS"))
            self.logger.info("Analyzing emotional tone of fables...")
            sentiment_results = self._run_sentiment_analysis(fables_by_language)
            results['sentiment'] = wrap_analysis_result(
                sentiment_results, "Sentiment", self.logger)

        # Style analysis
        if 'style' in analysis_types:
            self.logger.info(subsection_header("STYLE ANALYSIS"))
            self.logger.info("Analyzing writing style features...")
            style_results = self._run_style_analysis(fables_by_language)
            results['style'] = wrap_analysis_result(
                style_results, "Style", self.logger)

        # Syntax analysis
        if 'syntax' in analysis_types:
            self.logger.info(subsection_header("SYNTAX ANALYSIS"))
            self.logger.info("Analyzing grammatical structures...")
            syntax_results = self._run_syntax_analysis(fables_by_language)
            results['syntax'] = wrap_analysis_result(
                syntax_results, "Syntax", self.logger)
        
        # NLP techniques analysis
        if 'nlp_techniques' in analysis_types:
            self.logger.info(subsection_header("ADVANCED NLP TECHNIQUES"))
            self.logger.info("Applying TF-IDF, topic modeling, and word embeddings...")
            nlp_results = self._run_nlp_techniques_analysis(fables_by_language)
            results['nlp_techniques'] = wrap_analysis_result(
                nlp_results, "NLP Techniques", self.logger)
        
        # Statistical analysis
        if 'stats' in analysis_types:
            self.logger.info(subsection_header("STATISTICAL ANALYSIS"))
            self.logger.info("Computing statistical measures of text...")
            stats_results = self._run_stats_analysis(fables_by_language)
            results['stats'] = wrap_analysis_result(
                stats_results, "Statistics", self.logger)
        
        # Moral analysis
        if 'moral' in analysis_types:
            self.logger.info(subsection_header("MORAL ANALYSIS"))
            self.logger.info("Analyzing moral themes and lessons...")
            moral_results = self._run_moral_analysis(fables_by_language)
            results['moral'] = wrap_analysis_result(
                moral_results, "Moral", self.logger)
            
        # Cross-language analysis
        if 'cross_language' in analysis_types:
            self.logger.info(subsection_header("CROSS-LANGUAGE ANALYSIS"))
            self.logger.info("Comparing fables across different languages...")
            cross_lang_results = self._run_cross_language_analysis(fables_by_language)
            results['cross_language'] = wrap_analysis_result(
                cross_lang_results, "Cross-language", self.logger)
        
        self.logger.info(section_header("ANALYSIS SUMMARY"))
        completed_analyses = [k for k, v in results.items() 
                             if isinstance(v, dict) and 'error' not in v]
        self.logger.info("Successfully completed %d analyses", len(completed_analyses))
        self.logger.info("Analyzed %d fables across %d languages", 
                        sum(len(fables) for fables in fables_by_language.values()), 
                        len(fables_by_language))
        
        return results

    def _load_processed_fables(self, languages: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load processed fables for all languages with validation.
        
        Args:
            languages: List of language codes
            
        Returns:
            Dictionary mapping language codes to fable lists
        """
        # Clear existing data structures
        self.fables_by_language = {}
        self.fables_by_id = {}
        
        # Process each language
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
                        # Filter out non-dictionary items and ensure each has a valid ID
                        valid_fables = []
                        for i, fable in enumerate(fables):
                            if not isinstance(fable, dict):
                                self.logger.warning("Skipping invalid fable in %s: expected dict, got %s", 
                                                  lang, type(fable))
                                continue
                            
                            # Check if fable has a valid ID, assign one if not
                            if 'id' not in fable or not fable['id']:
                                # Generate a unique ID using index - this ensures we don't have empty IDs
                                fable['id'] = f"{lang}_{i+1}"
                                self.logger.warning("Assigned generated ID '%s' to fable in %s", 
                                                  fable['id'], lang)
                            
                            # Ensure language field exists
                            if 'language' not in fable:
                                fable['language'] = lang
                                
                            valid_fables.append(fable)
                        
                        self.fables_by_language[lang] = valid_fables
                        self.logger.info("Loaded %d valid fables for %s", len(valid_fables), lang)
                    else:
                        self.logger.warning("Data for %s is not in expected list format", lang)
                        
            except FileNotFoundError:
                self.logger.error("File not found: %s", processed_file)
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON in %s: %s", processed_file.name, e)
            except IOError as e:
                self.logger.error("I/O error reading %s: %s", processed_file, e)
            except Exception as e:
                self.logger.error("Unexpected error loading processed data for %s: %s (%s)", 
                                 lang, e, type(e).__name__)
        
        # Prepare fables_by_id dictionary with validation
        for lang, fables in self.fables_by_language.items():
            for fable in fables:
                fable_id = fable['id']  # We ensured this exists above
                if fable_id not in self.fables_by_id:
                    self.fables_by_id[fable_id] = {}
                self.fables_by_id[fable_id][lang] = fable
        
        return self.fables_by_language

    def _run_basic_analysis(self, fables_by_language: Dict[str, List[Dict[str, Any]]], 
                           analysis_types: List[str]) -> Dict[str, Any]:
        """
        Run the original basic analysis types with improved logging.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            analysis_types: List of analysis types to run
            
        Returns:
            Dictionary of analysis results
        """
        results = {}
        
        # POS tag distribution analysis
        if 'pos' in analysis_types:
            self.logger.info("Analyzing part-of-speech distributions...")
            pos_results = {}
            for lang in fables_by_language.keys():
                try:
                    self.logger.info("  Processing %s POS tags...", lang)
                    pos_dist = self.pos_analyzer.analyze_pos_distribution(lang)
                    pos_results[lang] = pos_dist
                    
                    # Add more detailed logging about the results
                    if pos_dist:
                        top_tags = sorted(pos_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                        top_tags_str = ", ".join([f"{tag}: {pct:.1f}%" for tag, pct in top_tags])
                        self.logger.info("  %s: Top POS tags are %s", lang, top_tags_str)
                except ValueError as e:
                    self.logger.error("Value error in POS analysis for %s: %s", lang, e)
                except KeyError as e:
                    self.logger.error("Key error in POS analysis for %s: %s", lang, e)
                except Exception as e:
                    self.logger.error("Error in POS analysis for %s: %s (%s)", 
                                     lang, e, type(e).__name__)
            
            results['pos_distribution'] = pos_results
            
            # Save analysis results
            for lang, dist in pos_results.items():
                if dist:
                    try:
                        self.writer.save_analysis_results(dist, lang, 'pos')
                    except IOError as e:
                        self.logger.error("Failed to save POS analysis for %s: %s", lang, e)
        
        # Entity analysis
        if 'entity' in analysis_types:
            self.logger.info("Analyzing named entity distributions...")
            entity_results = {}
            for lang in fables_by_language.keys():
                try:
                    entity_dist = self.entity_analyzer.analyze_entity_distribution(lang)
                    entity_results[lang] = entity_dist
                    
                    # Log entity distribution summary
                    if entity_dist:
                        entity_count = sum(1 for entity in entity_dist.values() 
                                         if isinstance(entity, dict) and 'count' in entity)
                        self.logger.info("  %s: Found %d entity types", lang, entity_count)
                except ValueError as e:
                    self.logger.error("Value error in entity analysis for %s: %s", lang, e)
                except KeyError as e:
                    self.logger.error("Key error in entity analysis for %s: %s", lang, e)
                except Exception as e:
                    self.logger.error("Error in entity analysis for %s: %s (%s)", 
                                     lang, e, type(e).__name__)
            
            results['entity_distribution'] = entity_results
            
            # Save analysis results
            for lang, dist in entity_results.items():
                if dist:
                    try:
                        self.writer.save_analysis_results(dist, lang, 'entity')
                    except IOError as e:
                        self.logger.error("Failed to save entity analysis for %s: %s", lang, e)
        
        # Cross-language fable comparison
        if 'comparison' in analysis_types:
            self.logger.info("Performing cross-language comparisons...")
            comparison_results = {}
            
            try:
                # Identify fable IDs that appear in multiple languages
                fable_ids = self.comparison_analyzer.find_common_fable_ids(fables_by_language)
                self.logger.info("Found %d fables with content in multiple languages", len(fable_ids))
                
                for i, fable_id in enumerate(fable_ids):
                    try:
                        self.logger.info("  Comparing fable %s [%d/%d]...", fable_id, i+1, len(fable_ids))
                        comparison = self.comparison_analyzer.compare_fable(fable_id, self.fables_by_id)
                        if comparison:
                            languages = comparison.get('languages', [])
                            self.logger.info("    Available in %d languages: %s", len(languages), ', '.join(languages))
                            comparison_results[fable_id] = comparison
                            
                            try:
                                self.writer.save_comparison_results(comparison, fable_id)
                            except IOError as e:
                                self.logger.error("Failed to save comparison for fable %s: %s", fable_id, e)
                    except ValueError as e:
                        self.logger.error("Value error comparing fable %s: %s", fable_id, e)
                    except KeyError as e:
                        self.logger.error("Key error comparing fable %s: %s", fable_id, e)
                    except Exception as e:
                        self.logger.error("Error comparing fable %s: %s (%s)", 
                                         fable_id, e, type(e).__name__)
            except Exception as e:
                self.logger.error("Error in cross-language comparison: %s (%s)", 
                                 e, type(e).__name__)
            
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
            
            self.logger.info("Preparing to cluster %d fables...", len(all_fables))
            
            # Run K-means clustering
            try:
                self.logger.info("Running K-means clustering...")
                kmeans_results = self.clustering_analyzer.kmeans_clustering(
                    all_fables, 
                    n_clusters=min(5, len(all_fables) // 2) if len(all_fables) > 5 else 2,
                    feature_type='tfidf'
                )
            except ValueError as e:
                self.logger.error("Value error in K-means clustering: %s", e)
                kmeans_results = {'error': f"K-means clustering failed: {e}"}
            except Exception as e:
                self.logger.error("Error in K-means clustering: %s", e)
                kmeans_results = {'error': f"K-means clustering failed: {e}"}
            
            # Run hierarchical clustering
            try:
                self.logger.info("Running hierarchical clustering...")
                hierarchical_results = self.clustering_analyzer.hierarchical_clustering(
                    all_fables,
                    feature_type='tfidf'
                )
            except ValueError as e:
                self.logger.error("Value error in hierarchical clustering: %s", e)
                hierarchical_results = {'error': f"Hierarchical clustering failed: {e}"}
            except Exception as e:
                self.logger.error("Error in hierarchical clustering: %s", e)
                hierarchical_results = {'error': f"Hierarchical clustering failed: {e}"}
            
            # Run DBSCAN clustering
            try:
                self.logger.info("Running DBSCAN clustering...")
                dbscan_results = self.clustering_analyzer.dbscan_clustering(
                    all_fables,
                    feature_type='tfidf'
                )
            except ValueError as e:
                self.logger.error("Value error in DBSCAN clustering: %s", e)
                dbscan_results = {'error': f"DBSCAN clustering failed: {e}"}
            except Exception as e:
                self.logger.error("Error in DBSCAN clustering: %s", e)
                dbscan_results = {'error': f"DBSCAN clustering failed: {e}"}
            
            # Cross-language clustering
            try:
                self.logger.info("Running cross-language clustering...")
                cross_lang_results = self.clustering_analyzer.cross_language_clustering(
                    self.fables_by_id,
                    feature_type='tfidf'
                )
            except ValueError as e:
                self.logger.error("Value error in cross-language clustering: %s", e)
                cross_lang_results = {'error': f"Cross-language clustering failed: {e}"}
            except Exception as e:
                self.logger.error("Error in cross-language clustering: %s", e)
                cross_lang_results = {'error': f"Cross-language clustering failed: {e}"}
            
            # Determine optimal number of clusters
            try:
                self.logger.info("Determining optimal number of clusters...")
                optimization_results = self.clustering_analyzer.optimize_clusters(
                    all_fables,
                    feature_type='tfidf'
                )
            except ValueError as e:
                self.logger.error("Value error in cluster optimization: %s", e)
                optimization_results = {'error': f"Cluster optimization failed: {e}"}
            except Exception as e:
                self.logger.error("Error in cluster optimization: %s", e)
                optimization_results = {'error': f"Cluster optimization failed: {e}"}
            
            # Combine results
            results = {
                'kmeans': kmeans_results,
                'hierarchical': hierarchical_results,
                'dbscan': dbscan_results,
                'cross_language': cross_lang_results,
                'optimal_clusters': optimization_results
            }
            
            # Save results
            try:
                self.clustering_analyzer.save_analysis('all', 'kmeans', kmeans_results)
                self.clustering_analyzer.save_analysis('all', 'hierarchical', hierarchical_results)
                self.clustering_analyzer.save_analysis('all', 'dbscan', dbscan_results)
                self.clustering_analyzer.save_analysis('all', 'cross_language', cross_lang_results)
            except IOError as e:
                self.logger.error("Failed to save clustering results: %s", e)
            
            return results
            
        except ValueError as e:
            self.logger.error("Value error in clustering analysis: %s", e)
            return {'error': f"Clustering analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in clustering analysis: %s", e)
            return {'error': f"Clustering analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in clustering analysis: %s", e)
            return {'error': f"Clustering analysis failed: {e}"}

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
                self.logger.info("Analyzing sentiment for %s fables...", lang)
                lang_results = []
                
                for i, fable in enumerate(fables):
                    try:
                        self.logger.debug("  Processing fable %d/%d: %s", i+1, len(fables), fable.get('title', 'Untitled'))
                        sentiment_result = self.sentiment_analyzer.analyze_sentiment(fable)
                        
                        # Add fable ID for reference
                        fable_id = fable.get('id', 'unknown')
                        sentiment_result['fable_id'] = fable_id
                        
                        lang_results.append(sentiment_result)
                    except ValueError as e:
                        self.logger.error("Value error analyzing sentiment for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except KeyError as e:
                        self.logger.error("Key error analyzing sentiment for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except Exception as e:
                        self.logger.error("Error analyzing sentiment for fable %s: %s (%s)", 
                                         fable.get('id', f"{lang}_{i}"), e, type(e).__name__)
                
                sentiment_by_language[lang] = lang_results
                self.logger.info("Completed sentiment analysis for %d %s fables", len(lang_results), lang)
            
            # Compare sentiment across languages for the same fables
            try:
                self.logger.info("Comparing sentiment across languages...")
                sentiment_comparison = self.sentiment_analyzer.compare_sentiment_across_languages(
                    self.fables_by_id
                )
            except ValueError as e:
                self.logger.error("Value error in cross-language sentiment comparison: %s", e)
                sentiment_comparison = {'error': f"Sentiment comparison failed: {e}"}
            except Exception as e:
                self.logger.error("Error in cross-language sentiment comparison: %s", e)
                sentiment_comparison = {'error': f"Sentiment comparison failed: {e}"}
            
            # Correlate sentiment with moral type
            try:
                self.logger.info("Correlating sentiment with moral types...")
                all_fables_with_sentiment = []
                for lang_results in sentiment_by_language.values():
                    all_fables_with_sentiment.extend(lang_results)
                
                sentiment_moral_correlation = self.sentiment_analyzer.correlate_sentiment_with_moral_type(
                    all_fables_with_sentiment
                )
            except ValueError as e:
                self.logger.error("Value error in sentiment-moral correlation: %s", e)
                sentiment_moral_correlation = {'error': f"Sentiment-moral correlation failed: {e}"}
            except Exception as e:
                self.logger.error("Error in sentiment-moral correlation: %s", e)
                sentiment_moral_correlation = {'error': f"Sentiment-moral correlation failed: {e}"}
            
            # Combine results
            results = {
                'by_language': sentiment_by_language,
                'cross_language_comparison': sentiment_comparison,
                'moral_correlation': sentiment_moral_correlation
            }
            
            return results
            
        except ValueError as e:
            self.logger.error("Value error in sentiment analysis: %s", e)
            return {'error': f"Sentiment analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in sentiment analysis: %s", e)
            return {'error': f"Sentiment analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in sentiment analysis: %s", e)
            return {'error': f"Sentiment analysis failed: {e}"}

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
                self.logger.info("Analyzing writing style for %s fables...", lang)
                lang_results = []
                
                for i, fable in enumerate(fables):
                    try:
                        # Get fable ID for reference
                        fable_id = fable.get('id', 'unknown')
                        title = fable.get('title', 'Untitled')
                        self.logger.debug("  Processing fable %d/%d: %s", i+1, len(fables), title)
                        
                        # Analyze sentence complexity
                        complexity_result = self.style_analyzer.sentence_complexity(fable)
                        
                        # Analyze lexical richness
                        richness_result = self.style_analyzer.lexical_richness(fable)
                        
                        # Analyze rhetorical devices
                        devices_result = self.style_analyzer.rhetorical_devices(fable)
                        
                        # Combine results for this fable
                        fable_result = {
                            'fable_id': fable_id,
                            'title': title,
                            'sentence_complexity': complexity_result,
                            'lexical_richness': richness_result,
                            'rhetorical_devices': devices_result
                        }
                        
                        lang_results.append(fable_result)
                    except ValueError as e:
                        self.logger.error("Value error in style analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except KeyError as e:
                        self.logger.error("Key error in style analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except Exception as e:
                        self.logger.error("Error in style analysis for fable %s: %s (%s)", 
                                         fable.get('id', f"{lang}_{i}"), e, type(e).__name__)
                
                style_by_language[lang] = lang_results
                self.logger.info("Completed style analysis for %d %s fables", len(lang_results), lang)
            
            return style_by_language
            
        except ValueError as e:
            self.logger.error("Value error in style analysis: %s", e)
            return {'error': f"Style analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in style analysis: %s", e)
            return {'error': f"Style analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in style analysis: %s", e)
            return {'error': f"Style analysis failed: {e}"}

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
                self.logger.info("Analyzing syntactic structures for %s fables...", lang)
                lang_results = []
                
                for i, fable in enumerate(fables):
                    try:
                        # Get fable ID for reference
                        fable_id = fable.get('id', 'unknown')
                        title = fable.get('title', 'Untitled')
                        self.logger.debug("  Processing fable %d/%d: %s", i+1, len(fables), title)
                        
                        # Analyze dependency frequencies
                        self.logger.debug("    Analyzing dependency frequencies...")
                        dep_freq_result = self.syntax_analyzer.dependency_frequencies(fable)
                        
                        # Analyze dependency distances
                        self.logger.debug("    Analyzing dependency distances...")
                        dep_dist_result = self.syntax_analyzer.dependency_distances(fable)
                        
                        # Analyze tree shapes
                        self.logger.debug("    Analyzing syntactic tree shapes...")
                        tree_result = self.syntax_analyzer.tree_shapes(fable)
                        
                        # Analyze dominant constructions
                        self.logger.debug("    Analyzing dominant constructions...")
                        const_result = self.syntax_analyzer.dominant_constructions(fable)
                        
                        # Analyze semantic roles
                        self.logger.debug("    Analyzing semantic roles...")
                        roles_result = self.syntax_analyzer.semantic_roles(fable)
                        
                        # Combine results for this fable
                        fable_result = {
                            'fable_id': fable_id,
                            'title': title,
                            'dependency_frequencies': dep_freq_result,
                            'dependency_distances': dep_dist_result,
                            'tree_shapes': tree_result,
                            'dominant_constructions': const_result,
                            'semantic_roles': roles_result
                        }
                        
                        # Save analysis results
                        try:
                            self.syntax_analyzer.save_analysis(fable_id, lang, 'dependency_frequencies', dep_freq_result)
                            self.syntax_analyzer.save_analysis(fable_id, lang, 'tree_shapes', tree_result)
                        except IOError as e:
                            self.logger.error("Failed to save syntax analysis for fable %s: %s", fable_id, e)
                        
                        lang_results.append(fable_result)
                    except ValueError as e:
                        self.logger.error("Value error in syntax analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except KeyError as e:
                        self.logger.error("Key error in syntax analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except Exception as e:
                        self.logger.error("Error in syntax analysis for fable %s: %s (%s)", 
                                         fable.get('id', f"{lang}_{i}"), e, type(e).__name__)
                
                syntax_by_language[lang] = lang_results
                self.logger.info("Completed syntax analysis for %d %s fables", len(lang_results), lang)
            
            # Run cross-fable comparison
            try:
                self.logger.info("Comparing syntactic structures across fables...")
                syntax_comparison = self.syntax_analyzer.compare_fables(
                    self.fables_by_id, 'dominant_constructions'
                )
            except ValueError as e:
                self.logger.error("Value error in syntax comparison: %s", e)
                syntax_comparison = {'error': f"Syntax comparison failed: {e}"}
            except KeyError as e:
                self.logger.error("Key error in syntax comparison: %s", e)
                syntax_comparison = {'error': f"Syntax comparison failed: {e}"}
            except Exception as e:
                self.logger.error("Error in syntax comparison: %s", e)
                syntax_comparison = {'error': f"Syntax comparison failed: {e}"}
            
            # Combine results
            results = {
                'by_language': syntax_by_language,
                'cross_fable_comparison': syntax_comparison
            }
            
            return results
            
        except ValueError as e:
            self.logger.error("Value error in syntax analysis: %s", e)
            return {'error': f"Syntax analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in syntax analysis: %s", e)
            return {'error': f"Syntax analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in syntax analysis: %s", e)
            return {'error': f"Syntax analysis failed: {e}"}

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
            try:
                self.logger.info("Running TF-IDF analysis across all fables...")
                tfidf_results = self.nlp_techniques.tfidf_analysis(fables_by_language)
            except ValueError as e:
                self.logger.error("Value error in TF-IDF analysis: %s", e)
                tfidf_results = {'error': f"TF-IDF analysis failed: {e}"}
            except Exception as e:
                self.logger.error("Error in TF-IDF analysis: %s", e)
                tfidf_results = {'error': f"TF-IDF analysis failed: {e}"}
            
            # Run topic modeling
            try:
                self.logger.info("Performing topic modeling with LDA...")
                topic_results = self.nlp_techniques.topic_modeling(
                    fables_by_language, 
                    n_topics=5,
                    method='lda'
                )
            except ValueError as e:
                self.logger.error("Value error in topic modeling: %s", e)
                topic_results = {'error': f"Topic modeling failed: {e}"}
            except Exception as e:
                self.logger.error("Error in topic modeling: %s", e)
                topic_results = {'error': f"Topic modeling failed: {e}"}
            
            # Run word embeddings analysis
            try:
                self.logger.info("Generating word embeddings...")
                embedding_results = self.nlp_techniques.word_embeddings(
                    fables_by_language,
                    model_type='word2vec'
                )
            except ValueError as e:
                self.logger.error("Value error in word embeddings: %s", e)
                embedding_results = {'error': f"Word embeddings failed: {e}"}
            except Exception as e:
                self.logger.error("Error in word embeddings: %s", e)
                embedding_results = {'error': f"Word embeddings failed: {e}"}
            
            # Combine results
            results = {
                'tfidf_analysis': tfidf_results,
                'topic_modeling': topic_results,
                'word_embeddings': embedding_results
            }
            
            # Save results
            try:
                self.nlp_techniques.save_analysis('all', 'tfidf', tfidf_results)
                self.nlp_techniques.save_analysis('all', 'topic_modeling', topic_results)
            except IOError as e:
                self.logger.error("Failed to save NLP techniques results: %s", e)
            
            return results
            
        except ValueError as e:
            self.logger.error("Value error in NLP techniques analysis: %s", e)
            return {'error': f"NLP techniques analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in NLP techniques analysis: %s", e)
            return {'error': f"NLP techniques analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in NLP techniques analysis: %s", e)
            return {'error': f"NLP techniques analysis failed: {e}"}

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
            self.logger.info("Analyzing word frequencies...")
            for lang, fables in fables_by_language.items():
                self.logger.info("Processing %s fables...", lang)
                lang_results = []
                
                for i, fable in enumerate(fables):
                    try:
                        # Get fable ID for reference
                        fable_id = fable.get('id', 'unknown')
                        title = fable.get('title', 'Untitled')
                        self.logger.debug("  Processing fable %d/%d: %s", i+1, len(fables), title)
                        
                        # Analyze word frequency
                        freq_result = self.stats_analyzer.word_frequency(fable)
                        
                        # Add fable ID
                        freq_result['fable_id'] = fable_id
                        freq_result['title'] = title
                        
                        lang_results.append(freq_result)
                        
                        # Save results
                        try:
                            self.stats_analyzer.save_analysis(fable_id, 'word_frequency', freq_result)
                        except IOError as e:
                            self.logger.error("Failed to save word frequency analysis for fable %s: %s", fable_id, e)
                    except ValueError as e:
                        self.logger.error("Value error in word frequency analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except KeyError as e:
                        self.logger.error("Key error in word frequency analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except Exception as e:
                        self.logger.error("Error in word frequency analysis for fable %s: %s (%s)", 
                                         fable.get('id', f"{lang}_{i}"), e, type(e).__name__)
                
                word_freq_by_language[lang] = lang_results
                self.logger.info("Completed word frequency analysis for %d %s fables", len(lang_results), lang)
            
            # Run chi-square test for POS distribution
            try:
                self.logger.info("Running chi-square tests on POS distributions...")
                chi_square_results = self.stats_analyzer.chi_square_test(
                    fables_by_language, 
                    feature='pos'
                )
            except ValueError as e:
                self.logger.error("Value error in chi-square test: %s", e)
                chi_square_results = {'error': f"Chi-square test failed: {e}"}
            except Exception as e:
                self.logger.error("Error in chi-square test: %s", e)
                chi_square_results = {'error': f"Chi-square test failed: {e}"}
            
            # Compare lexical diversity across languages
            try:
                self.logger.info("Comparing lexical diversity across languages...")
                lexical_diversity = {}
                for fable_id, lang_fables in self.fables_by_id.items():
                    if len(lang_fables) >= 2:  # Only compare if available in multiple languages
                        try:
                            diversity_result = self.stats_analyzer.compare_lexical_diversity(lang_fables)
                            lexical_diversity[fable_id] = diversity_result
                        except Exception as e:
                            self.logger.error("Error comparing lexical diversity for fable %s: %s", fable_id, e)
            except Exception as e:
                self.logger.error("Error in lexical diversity comparison: %s", e)
                lexical_diversity = {'error': f"Lexical diversity comparison failed: {e}"}
            
            # Combine results
            results = {
                'word_frequency': word_freq_by_language,
                'chi_square_test': chi_square_results,
                'lexical_diversity': lexical_diversity
            }
            
            return results
            
        except ValueError as e:
            self.logger.error("Value error in statistical analysis: %s", e)
            return {'error': f"Statistical analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in statistical analysis: %s", e)
            return {'error': f"Statistical analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in statistical analysis: %s", e)
            return {'error': f"Statistical analysis failed: {e}"}
            
    def _run_moral_analysis(self, fables_by_language):
        """
        Run moral analysis using the MoralDetector.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            Dictionary of moral analysis results
        """
        try:
            # Initialize results containers
            explicit_morals = {}
            implicit_morals = {}
            moral_themes = {}
            
            # Process each language
            for lang, fables in fables_by_language.items():
                self.logger.info("Analyzing morals in %s fables...", lang)
                lang_explicit = []
                lang_implicit = []
                lang_themes = []
                
                for i, fable in enumerate(fables):
                    # Skip if not a dictionary
                    if not isinstance(fable, dict):
                        self.logger.warning("Skipping non-dictionary fable in %s", lang)
                        continue
                    
                    try:
                        fable_id = fable.get('id', 'unknown')
                        title = fable.get('title', 'Untitled')
                        self.logger.debug("  Processing fable %d/%d: %s", i+1, len(fables), title)
                        
                        # Detect explicit moral
                        self.logger.debug("    Detecting explicit moral...")
                        explicit_result = self.moral_detector.detect_explicit_moral(fable)
                        if explicit_result.get('has_explicit_moral'):
                            explicit_result['fable_id'] = fable_id
                            explicit_result['title'] = title
                            lang_explicit.append(explicit_result)
                            self.logger.debug("    Found explicit moral")
                        
                        # Infer implicit moral
                        self.logger.debug("    Inferring implicit moral...")
                        implicit_result = self.moral_detector.infer_implicit_moral(fable, explicit_result)
                        if implicit_result.get('has_inferred_moral'):
                            implicit_result['fable_id'] = fable_id
                            implicit_result['title'] = title
                            lang_implicit.append(implicit_result)
                            self.logger.debug("    Inferred implicit moral")
                        
                        # Classify moral theme
                        self.logger.debug("    Classifying moral theme...")
                        moral_text = None
                        if explicit_result.get('has_explicit_moral'):
                            moral_text = explicit_result.get('moral_text')
                        elif implicit_result.get('has_inferred_moral'):
                            inferred = implicit_result.get('inferred_morals', [])
                            if inferred:
                                moral_text = inferred[0].get('text')
                        
                        if moral_text:
                            theme_result = self.moral_detector.classify_moral_theme(moral_text, lang)
                            theme_result['fable_id'] = fable_id
                            theme_result['title'] = title
                            theme_result['moral_text'] = moral_text
                            lang_themes.append(theme_result)
                            self.logger.debug("    Classified moral theme")
                    except ValueError as e:
                        self.logger.error("Value error in moral analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except KeyError as e:
                        self.logger.error("Key error in moral analysis for fable %s: %s", 
                                         fable.get('id', f"{lang}_{i}"), e)
                    except Exception as e:
                        self.logger.error("Error in moral analysis for fable %s: %s (%s)", 
                                         fable.get('id', f"{lang}_{i}"), e, type(e).__name__)
                
                # Store results for this language
                explicit_morals[lang] = lang_explicit
                implicit_morals[lang] = lang_implicit
                moral_themes[lang] = lang_themes
                
                self.logger.info("Found %d explicit and %d implicit morals in %s fables", 
                                len(lang_explicit), len(lang_implicit), lang)
            
            # Cross-language moral comparison 
            try:
                self.logger.info("Comparing morals across languages...")
                moral_comparison = self.moral_detector.compare_morals(self.fables_by_id)
            except ValueError as e:
                self.logger.error("Value error in moral comparison: %s", e)
                moral_comparison = {'error': f"Moral comparison failed: {e}"}
            except KeyError as e:
                self.logger.error("Key error in moral comparison: %s", e)
                moral_comparison = {'error': f"Moral comparison failed: {e}"}
            except Exception as e:
                self.logger.error("Error in moral comparison: %s", e)
                moral_comparison = {'error': f"Moral comparison failed: {e}"}
            
            # Combine results
            results = {
                'explicit_morals': explicit_morals,
                'implicit_morals': implicit_morals,
                'moral_themes': moral_themes,
                'cross_language_comparison': moral_comparison
            }
            
            # Save results
            try:
                for lang, morals in explicit_morals.items():
                    if morals:
                        self.writer.save_analysis_results(morals, lang, 'explicit_morals')
                
                for lang, morals in implicit_morals.items():
                    if morals:
                        self.writer.save_analysis_results(morals, lang, 'implicit_morals')
                
                for lang, themes in moral_themes.items():
                    if themes:
                        self.writer.save_analysis_results(themes, lang, 'moral_themes')
                
                # Save cross-language comparison
                if moral_comparison:
                    self.writer.save_analysis_results(moral_comparison, 'all', 'moral_comparison')
            except IOError as e:
                self.logger.error("Failed to save moral analysis results: %s", e)
            
            return results
            
        except ValueError as e:
            self.logger.error("Value error in moral analysis: %s", e)
            return {'error': f"Moral analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in moral analysis: %s", e)
            return {'error': f"Moral analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in moral analysis: %s", e)
            return {'error': f"Moral analysis failed: {e}"}
            
    def _run_cross_language_analysis(self, fables_by_language):
        """
        Run cross-language analysis comparing the same fable across languages.
        """
        try:
            # Check if fables_by_id is properly initialized
            if not hasattr(self, 'fables_by_id') or not self.fables_by_id:
                # Rebuild fables_by_id dictionary if needed
                self.logger.info("Rebuilding fables_by_id dictionary for cross-language analysis")
                self.fables_by_id = {}
                for lang, fables in fables_by_language.items():
                    for fable in fables:
                        if isinstance(fable, dict) and 'id' in fable:
                            fable_id = fable['id']
                            if fable_id not in self.fables_by_id:
                                self.fables_by_id[fable_id] = {}
                            self.fables_by_id[fable_id][lang] = fable

            self.logger.debug("fables_by_id structure contains %d unique fable IDs", len(self.fables_by_id))
            
            # Initialize result containers
            moral_comparison = {}
            entity_results = {}
            word_usage_comparison = {}

            # 1. Use the MoralDetector to compare morals across languages
            self.logger.info("Comparing moral themes across languages...")
            if isinstance(self.fables_by_id, dict):
                try:
                    moral_comparison = self.moral_detector.compare_morals(self.fables_by_id)
                    self.logger.info("Successfully compared morals across languages")
                except ValueError as moral_e:
                    self.logger.error("Value error in moral comparison: %s", moral_e)
                    moral_comparison = {"error": f"Moral comparison failed: {moral_e}"}
                except KeyError as moral_e:
                    self.logger.error("Key error in moral comparison: %s", moral_e)
                    moral_comparison = {"error": f"Moral comparison failed: {moral_e}"}
                except Exception as moral_e:
                    self.logger.error("Error in moral comparison: %s", moral_e)
                    moral_comparison = {"error": f"Moral comparison failed: {moral_e}"}
            else:
                self.logger.warning("Cannot perform moral comparison: fables_by_id is not a dictionary")
                moral_comparison = {"error": "Invalid data structure for moral comparison"}
            
            # 2. Use the entity analyzer to compare entity distributions
            self.logger.info("Analyzing entity distributions across languages...")
            for lang in fables_by_language.keys():
                try:
                    entity_dist = self.entity_analyzer.analyze_entity_distribution(lang)
                    entity_results[lang] = entity_dist
                    if entity_dist:
                        entity_count = sum(1 for v in entity_dist.values() 
                                         if isinstance(v, dict) and 'count' in v)
                        self.logger.info("  %s: Found %d entity types", lang, entity_count)
                except ValueError as entity_e:
                    self.logger.error("Value error analyzing entity distribution for language %s: %s", lang, entity_e)
                    entity_results[lang] = {"error": f"Entity analysis failed: {entity_e}"}
                except KeyError as entity_e:
                    self.logger.error("Key error analyzing entity distribution for language %s: %s", lang, entity_e)
                    entity_results[lang] = {"error": f"Entity analysis failed: {entity_e}"}
                except Exception as entity_e:
                    self.logger.error("Error analyzing entity distribution for language %s: %s", lang, entity_e)
                    entity_results[lang] = {"error": f"Entity analysis failed: {entity_e}"}
            
            # 3. Compare word usage across languages
            self.logger.info("Comparing word usage patterns across languages...")
            for fable_id, lang_fables in self.fables_by_id.items():
                # Skip if not a dictionary or not enough languages
                if not isinstance(lang_fables, dict) or len(lang_fables) < 2:
                    continue
                    
                # Make sure all items in lang_fables are also dictionaries
                all_dicts = all(isinstance(fable, dict) for fable in lang_fables.values())
                if not all_dicts:
                    self.logger.warning(f"Skipping word usage comparison for fable %s: not all language entries are dictionaries", fable_id)
                    continue
                    
                try:
                    usage_result = self.stats_analyzer.compare_word_usage(lang_fables)
                    word_usage_comparison[fable_id] = usage_result
                    self.logger.debug("Compared word usage for fable %s across %d languages", fable_id, len(lang_fables))
                except ValueError as word_e:
                    self.logger.error("Value error comparing word usage for fable %s: %s", fable_id, word_e)
                except KeyError as word_e:
                    self.logger.error("Key error comparing word usage for fable %s: %s", fable_id, word_e)
                except Exception as word_e:
                    self.logger.error("Error comparing word usage for fable %s: %s", fable_id, word_e)
                    # Continue with other fables rather than failing entire analysis
            
            # Combine results
            results = {
                'moral_comparison': moral_comparison,
                'entity_distribution': entity_results,
                'word_usage_comparison': word_usage_comparison
            }
            
            return results
            
        except ValueError as e:
            self.logger.error("Value error in cross-language analysis: %s", e)
            return {'error': f"Cross-language analysis failed with value error: {e}"}
        except KeyError as e:
            self.logger.error("Key error in cross-language analysis: %s", e)
            return {'error': f"Cross-language analysis failed with key error: {e}"}
        except Exception as e:
            self.logger.error("Error in cross-language analysis: %s", e)
            return {'error': f"Cross-language analysis failed: {e}"}