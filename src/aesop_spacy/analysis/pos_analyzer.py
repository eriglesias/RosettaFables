"""
Analyzes part-of-speech distributions in fables across languages.

This module provides:
- POS tag frequency distribution analysis
- POS pattern identification
- Cross-language POS comparison
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

class POSAnalyzer:
    """Analyzes part-of-speech patterns in multilingual fables."""
    
    def __init__(self, analysis_dir: Path):
        """
        Initialize the POS analyzer with the analysis directory.
        
        Args:
            analysis_dir: Directory for storing/retrieving analysis results
        """
        self.analysis_dir = analysis_dir
        self.logger = logging.getLogger(__name__)
    
    def analyze_pos_distribution(self, language: str) -> Dict[str, float]:
        """
        Analyze part-of-speech distribution for a language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with POS tag frequencies as percentages
        """
        try:
            # Load processed fables
            processed_file = self.analysis_dir.parent / "processed" / f"fables_{language}.json"
            
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
                pos_distribution = {
                    pos: (count / total_tokens * 100) 
                    for pos, count in pos_counts.items()
                }
                
                # Sort by frequency (descending)
                pos_distribution = dict(sorted(
                    pos_distribution.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
                
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
    
    def analyze_pos_patterns(self, fables: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        """
        Analyze common POS patterns and sequences in fables.
        
        Args:
            fables: List of fable dictionaries
            language: Language code
            
        Returns:
            Dictionary with POS pattern analysis
        """
        # Initialize result structure
        result = {
            'language': language,
            'bigram_patterns': Counter(),
            'trigram_patterns': Counter(),
            'sentence_initial_tags': Counter(),
            'sentence_final_tags': Counter(),
            'total_sentences': 0,
            'total_tokens': 0
        }
        
        # Process each fable
        for fable in fables:
            sentences = fable.get('sentences', [])
            
            for sentence in sentences:
                pos_tags = []
                
                # Extract POS tags
                if 'pos_tags' in sentence:
                    for tag_entry in sentence.get('pos_tags', []):
                        # Handle different formats of POS tags
                        if isinstance(tag_entry, list) and len(tag_entry) >= 2:
                            pos_tags.append(tag_entry[1])  # [token, pos_tag]
                        elif isinstance(tag_entry, dict) and 'pos' in tag_entry:
                            pos_tags.append(tag_entry['pos'])
                        elif isinstance(tag_entry, str):
                            pos_tags.append(tag_entry)
                
                if pos_tags:
                    # Count sentence-initial and sentence-final tags
                    result['sentence_initial_tags'][pos_tags[0]] += 1
                    result['sentence_final_tags'][pos_tags[-1]] += 1
                    result['total_sentences'] += 1
                    result['total_tokens'] += len(pos_tags)
                    
                    # Extract bigrams (pairs of consecutive POS tags)
                    for i in range(len(pos_tags) - 1):
                        bigram = (pos_tags[i], pos_tags[i+1])
                        result['bigram_patterns'][bigram] += 1
                    
                    # Extract trigrams (triplets of consecutive POS tags)
                    for i in range(len(pos_tags) - 2):
                        trigram = (pos_tags[i], pos_tags[i+1], pos_tags[i+2])
                        result['trigram_patterns'][trigram] += 1
        
        # Convert counters to serializable format
        result['bigram_patterns'] = {
            f"{big[0]}_{big[1]}": count 
            for big, count in result['bigram_patterns'].most_common(10)
        }
        
        result['trigram_patterns'] = {
            f"{tri[0]}_{tri[1]}_{tri[2]}": count 
            for tri, count in result['trigram_patterns'].most_common(10)
        }
        
        result['sentence_initial_tags'] = dict(result['sentence_initial_tags'].most_common(5))
        result['sentence_final_tags'] = dict(result['sentence_final_tags'].most_common(5))
        
        # Calculate frequencies
        if result['total_sentences'] > 0:
            result['sentence_initial_tags_freq'] = {
                tag: count / result['total_sentences'] * 100
                for tag, count in result['sentence_initial_tags'].items()
            }
            
            result['sentence_final_tags_freq'] = {
                tag: count / result['total_sentences'] * 100
                for tag, count in result['sentence_final_tags'].items()
            }
        
        return result
    
    def save_analysis(self, results: Dict[str, Any], language: str, analysis_type: str):
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results to save
            language: Language code
            analysis_type: Type of analysis (e.g., 'distribution', 'patterns')
        """
        # Create directory if it doesn't exist
        output_dir = self.analysis_dir / 'pos'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create filename
        filename = f"{language}_{analysis_type}.json"
        output_path = output_dir / filename
        
        # Save to JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved POS %s analysis for %s to %s", 
                            analysis_type, language, output_path)
        except Exception as e:
            self.logger.error("Error saving POS analysis: %s", e)