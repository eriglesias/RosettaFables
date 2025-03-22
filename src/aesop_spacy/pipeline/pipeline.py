# src/aesop_spacy/pipeline/pipeline.py
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from ..io.loader import FableLoader
from ..io.writer import OutputWriter
from ..io.serializer import SpacySerializer
from ..preprocessing.cleaner import TextCleaner
from ..preprocessing.extractor import ContentExtractor
from ..preprocessing.processor import FableProcessor
from ..models.model_manager import get_model


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
        self.writer = OutputWriter(output_dir)
        
        self.logger.info("Fable pipeline initialized")
        self.logger.info(f"Data directory: {data_dir}")
        self.logger.info(f"Output directory: {output_dir}")
    
    def run(self):
        """
        Run the complete pipeline from loading to processing to analysis.
        
        Returns:
            True if the pipeline completed successfully
        """
        self.logger.info("Starting fable processing pipeline")
        
        # Load all fables from both JSON and markdown files
        fables_by_language = self.loader.load_all()
        
        # Log what we found
        total_fables = sum(len(fables) for fables in fables_by_language.values())
        self.logger.info(f"Loaded {total_fables} fables across {len(fables_by_language)} languages")
        
        # Process each language
        for lang, fables in fables_by_language.items():
            self._process_language(lang, fables)
            
        self.logger.info("Pipeline execution completed successfully")
        return True
    
    def _process_language(self, language: str, fables: List[Dict[str, Any]]):
        """
        Process all fables for a specific language.
        
        Args:
            language: Language code
            fables: List of fables for this language
        """
        if not fables:
            self.logger.warning(f"No fables to process for language: {language}")
            return
            
        self.logger.info(f"Processing {len(fables)} fables for language: {language}")
        
        # Get the appropriate model
        model = get_model(language)
        if not model:
            self.logger.warning(f"Skipping {language} - no NLP model available")
            return
            
        self.logger.info(f"Using model {model.meta.get('name', 'unknown')} for {language}")
        
        # Process each fable
        processed_fables = []
        for fable in fables:
            try:
                # Clean the text
                cleaned_fable = self.cleaner.clean_fable(fable)
                self.logger.debug(f"Cleaned fable: {cleaned_fable.get('title', 'Untitled')}")
                
                # Extract content
                extracted_fable = self.extractor.extract_content(cleaned_fable)
                self.logger.debug(f"Extracted content for: {extracted_fable.get('title', 'Untitled')}")
                
                # Process with NLP
                processed_fable = self.processor.process_fable(extracted_fable, model)
                self.logger.debug(f"Processed fable: {processed_fable.get('title', 'Untitled')}")
                
                # Serialize spaCy objects to JSON-compatible format
                serialized_fable = self.serializer.serialize(processed_fable)
                
                processed_fables.append(serialized_fable)
                
            except Exception as e:
                self.logger.error(f"Error processing fable {fable.get('title', 'Untitled')}: {e}", exc_info=True)
        
        # Save results
        if processed_fables:
            output_file = self.writer.save_processed_fables(processed_fables, language)
            self.logger.info(f"Saved {len(processed_fables)} processed fables to {output_file}")
        
    def analyze(self, analysis_types=None):
        """
        Run analyses on processed fables.
        
        Args:
            analysis_types: List of analysis types to run, or None for all
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Starting fable analysis")
        
        # Default to all analysis types if none specified
        if analysis_types is None:
            analysis_types = ['pos', 'entity', 'moral', 'comparison']
            
        # Results container
        results = {}
        
        # Load processed fables for each language
        languages = []
        for lang_file in (self.output_dir / "processed").glob("fables_*.json"):
            lang = lang_file.stem.split('_')[1]
            languages.append(lang)
        
        self.logger.info(f"Found processed data for languages: {', '.join(languages)}")
        
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
        
        # Cross-language fable comparison
        if 'comparison' in analysis_types:
            comparison_results = {}
            
            # Identify fable IDs that appear in multiple languages
            fable_ids = self._find_common_fable_ids(languages)
            self.logger.info(f"Found {len(fable_ids)} fables with content in multiple languages")
            
            for fable_id in fable_ids:
                comparison = self._compare_fable(fable_id, languages)
                if comparison:
                    comparison_results[fable_id] = comparison
                    self.writer.save_comparison_results(comparison, fable_id)
            
            results['fable_comparisons'] = comparison_results
        
        self.logger.info("Analysis completed successfully")
        return results
    
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
                self.logger.warning(f"No processed data for {language}")
                return {}
                
            with open(processed_file, 'r', encoding='utf-8') as f:
                fables = json.load(f)
                
            # Count POS tags
            pos_counts = {}
            total_tokens = 0
            
            for fable in fables:
                for token, pos in fable.get('pos_tags', []):
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
                
                self.logger.info(f"Analyzed POS distribution for {language}: {len(pos_distribution)} tags from {total_tokens} tokens")
                return pos_distribution
            else:
                self.logger.warning(f"No tokens found for {language}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error analyzing POS distribution for {language}: {e}")
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
                self.logger.warning(f"No processed data for {language}")
                return {}
                
            with open(processed_file, 'r', encoding='utf-8') as f:
                fables = json.load(f)
                
            # Count entity types
            entity_counts = {}
            entity_examples = {}
            total_entities = 0
            
            for fable in fables:
                for entity, entity_type, *_ in fable.get('entities', []):
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
                
                self.logger.info(f"Analyzed entity distribution for {language}: {len(entity_distribution)} types from {total_entities} entities")
                return entity_distribution
            else:
                self.logger.warning(f"No entities found for {language}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error analyzing entity distribution for {language}: {e}")
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
                    fables = json.load(f)
                    fables_by_language[lang] = fables
            except Exception as e:
                self.logger.error(f"Error loading processed data for {lang}: {e}")
        
        # Find IDs that appear in multiple languages
        fable_ids_by_language = {}
        
        for lang, fables in fables_by_language.items():
            fable_ids = [fable.get('id') for fable in fables if fable.get('id')]
            fable_ids_by_language[lang] = set(fable_ids)
        
        # Find IDs that appear in at least 2 languages
        if not fable_ids_by_language:
            return []
            
        all_ids = set().union(*fable_ids_by_language.values())
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
            Comparison data dictionary
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
                    fables = json.load(f)
                    
                # Find the specific fable
                for fable in fables:
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
                        for _, pos in fable.get('pos_tags', []):
                            pos_counts[pos] = pos_counts.get(pos, 0) + 1
                            
                        total_tokens = sum(pos_counts.values())
                        if total_tokens > 0:
                            comparison['pos_distribution'][lang] = {
                                pos: count / total_tokens * 100 
                                for pos, count in pos_counts.items()
                            }
                        
                        # Check moral
                        has_moral = bool(fable.get('moral', {}).get('text', ''))
                        comparison['has_moral'][lang] = has_moral
                        
                        if has_moral:
                            moral_text = fable.get('moral', {}).get('text', '')
                            comparison['moral_length'][lang] = len(moral_text.split())
                        
                        break
                    
            except Exception as e:
                self.logger.error(f"Error comparing fable {fable_id} for {lang}: {e}")
        
        # Only return if we found the fable in at least 2 languages
        if len(comparison['languages']) >= 2:
            self.logger.info(f"Compared fable {fable_id} across {len(comparison['languages'])} languages")
            return comparison
        else:
            self.logger.warning(f"Fable {fable_id} not found in multiple languages")
            return None