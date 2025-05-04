# syntax_analyzer.py
"""
Analyzes syntactic structures across languages in fables.

This module provides analysis of:
- Dependency relation frequencies
- Dependency distances
- Tree shapes and structures
- Dominant syntactic constructions
- Semantic role patterns
"""

from pathlib import Path
from collections import Counter, defaultdict
import logging
import json

class SyntaxAnalyzer:
    """Analyzes syntax structures across languages in fables."""

    def __init__(self, analysis_dir):
        """
        Initialize the syntax analyzer.
        
        Args:
            analysis_dir: Directory for storing/retrieving analysis results
        """
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(__name__)


    def dependency_frequencies(self, fable):
        """
        Count frequencies of dependency relations in a fable.
        """
        # Get sentences from the fable
        sentences = fable.get('sentences', [])
        self.logger.info("Processing %d sentences for dependency frequency analysis", len(sentences))
        
        # Add diagnostic logging
        if sentences and len(sentences) > 0:
            self.logger.debug("Sample sentence keys: %s", list(sentences[0].keys()))
            if 'dependencies' in sentences[0]:
                self.logger.debug("Found %d dependencies in first sentence", 
                                len(sentences[0]['dependencies']))
            else:
                self.logger.warning("No dependencies key in first sentence")
        
        # Check if dependencies exist at document level
        if 'dependencies' in fable:
            self.logger.debug("Found %d dependencies at document level", len(fable['dependencies']))
        
        # Initialize counters
        dep_counts = {}
        dep_examples = {}
        total_deps = 0
        
        # First look for sentence-level dependencies
        for i, sentence in enumerate(sentences):
            # Get dependency information if available
            deps = sentence.get('dependencies', [])
            
            if not deps:
                self.logger.debug("Sentence %d has no dependencies", i)
                continue
                
            self.logger.debug("Sentence %d has %d dependencies", i, len(deps))
            
            # Process each dependency
            for dep in deps:
                # Extract dependency type, head word, and dependent word
                dep_type = dep.get('dep')
                head = dep.get('head_text', '')
                dependent = dep.get('dependent_text', '')
                
                if not dep_type:
                    self.logger.debug("Missing dependency type in: %s", dep)
                    continue
                    
                # Update counts
                dep_counts[dep_type] = dep_counts.get(dep_type, 0) + 1
                total_deps += 1
                
                # Store an example
                if dep_type not in dep_examples and head and dependent:
                    dep_examples[dep_type] = f"{dependent} → {head}"
        
        # If no sentence-level dependencies found, try document-level
        if total_deps == 0 and 'dependencies' in fable:
            self.logger.info("No sentence-level dependencies, trying document-level")
            doc_deps = fable.get('dependencies', [])
            
            for dep in doc_deps:
                dep_type = dep.get('dep')
                head = dep.get('head_text', '')
                dependent = dep.get('dependent_text', '')
                
                if dep_type:
                    dep_counts[dep_type] = dep_counts.get(dep_type, 0) + 1
                    total_deps += 1
                    
                    if dep_type not in dep_examples and head and dependent:
                        dep_examples[dep_type] = f"{dependent} → {head}"
        
        # If still no dependencies, check if we can recover from raw sentence text
        if total_deps == 0 and len(sentences) > 0:
            self.logger.warning("No dependencies found. This might be due to missing dependency parsing in the NLP pipeline.")
            self.logger.info("Consider reprocessing the fables with a model that includes dependency parsing.")
        
        self.logger.info("Found %d total dependencies across %d sentences", 
                        total_deps, len(sentences))
        
        # Calculate percentages
        results = {
            'total_dependencies': total_deps,
            'frequencies': {},
            'examples': dep_examples
        }
        
        # Convert counts to percentages
        if total_deps > 0:
            results['frequencies'] = {
                dep: (count / total_deps) * 100
                for dep, count in dep_counts.items()
            }
        
        return results

        
    def dependency_distances(self, fable):
        """
        Calculate average dependency distances (how far apart connected words are).
        
        Returns:
            Dict with average distances and distribution statistics
        """
        sentences = fable.get('sentences', [])
        
        # Track all distances
        all_distances = []
        distances_by_type = {}
        
        for sentence in sentences:
            self.logger.debug(f"Dependencies keys in sentence: {list(sentence.keys())}")
            if 'dependencies' in sentence and 'tokens' in sentence:
                self.logger.debug(f"Found {len(sentence['dependencies'])} dependencies")
                deps = sentence['dependencies']
                tokens = sentence['tokens']
                
                # Create a map of token indices
                token_indices = {}
                for i, token in enumerate(tokens):
                    token_id = token.get('id')
                    if token_id is not None:
                        token_indices[token_id] = i
                
                # Calculate distances
                for dep in deps:
                    head_id = dep.get('head_id')
                    dep_id = dep.get('dependent_id')
                    dep_type = dep.get('dep')
                    
                    # Calculate distance if we have both positions
                    if head_id in token_indices and dep_id in token_indices:
                        distance = abs(token_indices[head_id] - token_indices[dep_id])
                        
                        # Filter out unreasonably large distances (likely errors)
                        max_reasonable_distance = 30  # Most natural language dependencies stay under this
                        if distance <= max_reasonable_distance:
                            all_distances.append(distance)
                            
                            # Track by dependency type
                            if dep_type not in distances_by_type:
                                distances_by_type[dep_type] = []
                            distances_by_type[dep_type].append(distance)
        
        # Calculate statistics
        results = {
            'overall': {
                'average_distance': 0,
                'max_distance': 0,
                'min_distance': 0,
                'total_dependencies': len(all_distances)
            },
            'by_dependency_type': {}
        }
        
        # Overall statistics
        if all_distances:
            results['overall']['average_distance'] = sum(all_distances) / len(all_distances)
            results['overall']['max_distance'] = max(all_distances)
            results['overall']['min_distance'] = min(all_distances)
        
        # By dependency type
        for dep_type, distances in distances_by_type.items():
            if distances:
                results['by_dependency_type'][dep_type] = {
                    'average_distance': sum(distances) / len(distances),
                    'count': len(distances)
                }

        return results


    def tree_shapes(self, fable):
        """
        Analyze dependency tree structures.
        
        Measures:
        - Branching factor (how many dependents per word)
        - Tree width vs. depth ratio
        - Projectivity (crossing dependencies)
        
        Returns:
            Dict with tree shape metrics
        """
        sentences = fable.get('sentences', [])
        language = fable.get('language', 'en')

        results = {
            'average_branching_factor': 0,
            'max_branching_factor': 0,
            'width_depth_ratios': [],
            'average_width_depth_ratio': 0,
            'non_projective_count': 0,
            'sentence_count': len(sentences),
            'language_insights': {}
        }
        # Skip if no sentences
        if not sentences:
            return results
            
        # Analyze each sentence's tree
        branching_factors = []
        
        for sentence in sentences:
            if 'dependencies' in sentence and 'tokens' in sentence:
                deps = sentence['dependencies']
                tokens = sentence['tokens']
                
                # Create a dependency map (which tokens depend on which head)
                head_to_deps = defaultdict(list)
                token_positions = {}
                
                # Map token IDs to their positions in the sentence
                for i, token in enumerate(tokens):
                    token_id = token.get('id')
                    if token_id is not None:
                        token_positions[token_id] = i
                
                # Build dependency tree structure
                for dep in deps:
                    head_id = dep.get('head_id')
                    dep_id = dep.get('dependent_id')
                    
                    if head_id is not None and dep_id is not None:
                        head_to_deps[head_id].append(dep_id)
                
                # Calculate branching factor for each head
                head_branching = [len(dependents) for head, dependents in head_to_deps.items()]
                if head_branching:
                    avg_branching = sum(head_branching) / len(head_branching)
                    max_branching = max(head_branching)
                    branching_factors.append(avg_branching)
                    
                    if max_branching > results['max_branching_factor']:
                        results['max_branching_factor'] = max_branching
                
                # Find root node (token with head=0 or similar root convention)
                root_id = None
                for dep in deps:
                    if dep.get('dep') == 'ROOT' or dep.get('head_id') == 0:
                        root_id = dep.get('dependent_id')
                        break
                
                if root_id is not None:
                    # Calculate tree depth and width
                    max_depth, tree_width = self._calculate_tree_dimensions(root_id, head_to_deps)
                    
                    # Calculate width-to-depth ratio if depth > 0
                    if max_depth > 0:
                        width_depth_ratio = tree_width / max_depth
                        results['width_depth_ratios'].append(width_depth_ratio)
                
                # Check for crossing dependencies (non-projectivity)
                crossings = self._count_crossing_dependencies(deps, token_positions)
                if crossings > 0:
                    results['non_projective_count'] += 1
        
        # Calculate averages across all sentences
        if branching_factors:
            results['average_branching_factor'] = sum(branching_factors) / len(branching_factors)
        
        if results['width_depth_ratios']:
            results['average_width_depth_ratio'] = sum(results['width_depth_ratios']) / len(results['width_depth_ratios'])
        
        # Add language-specific insights
        language_insights = {
            'en': {
                'typical_branching': 'Right-branching dominant in English',
                'typical_width_depth': 'English tends toward wider, shallower trees than SOV languages'
            },
            'es': {
                'typical_branching': 'Right-branching dominant in Spanish',
                'typical_width_depth': 'Spanish allows slightly deeper trees due to flexible word order'
            },
            'de': {
                'typical_branching': 'German mixes left and right branching',
                'typical_width_depth': 'German verb-final structures create deeper trees'
            },
            'nl': {
                'typical_branching': 'Dutch mixes left and right branching like German',
                'typical_width_depth': 'Dutch verb-final structures create deeper trees'
            },
            'grc': {
                'typical_branching': 'Ancient Greek had flexible branching due to case marking',
                'typical_width_depth': 'Flexible word order allows for deeper trees'
            }
        }
        
        if language in language_insights:
            results['language_insights'] = language_insights[language]
        
        return results
    
    def _calculate_tree_dimensions(self, node_id, head_to_deps, current_depth=1):
        """
        Recursively calculate the depth and width of a tree.
        
        Args:
            node_id: Current node ID
            head_to_deps: Mapping of head IDs to dependent IDs
            current_depth: Current depth in the tree
            
        Returns:
            Tuple of (max_depth, tree_width)
        """
        if node_id not in head_to_deps or not head_to_deps[node_id]:
            # Leaf node
            return current_depth, 1
        
        children = head_to_deps[node_id]
        max_child_depth = current_depth
        total_width = 0
        
        for child_id in children:
            child_depth, child_width = self._calculate_tree_dimensions(
                child_id, head_to_deps, current_depth + 1
            )
            max_child_depth = max(max_child_depth, child_depth)
            total_width += child_width
        
        return max_child_depth, max(1, total_width)
    
    def _count_crossing_dependencies(self, deps, token_positions):
        """
        Count crossing dependencies (non-projective edges).
        
        A dependency is crossing if it creates a non-projective structure:
        When an arc A -> B crosses over another arc C -> D.
        
        Args:
            deps: List of dependency dictionaries
            token_positions: Mapping of token IDs to positions
            
        Returns:
            Number of crossing dependencies
        """
        crossings = 0
        
        # Only check if we have position information
        if not token_positions:
            return 0
        
        # Get all arcs (as position pairs)
        arcs = []
        for dep in deps:
            head_id = dep.get('head_id')
            dep_id = dep.get('dependent_id')
            
            if head_id in token_positions and dep_id in token_positions:
                head_pos = token_positions[head_id]
                dep_pos = token_positions[dep_id]
                arcs.append((min(head_pos, dep_pos), max(head_pos, dep_pos)))
        
        # Check every pair of arcs for crossings
        for i, (start1, end1) in enumerate(arcs):
            for start2, end2 in arcs[i+1:]:
                # Crossing condition: one arc starts inside another and ends outside
                if (start1 < start2 < end1 < end2) or (start2 < start1 < end2 < end1):
                    crossings += 1
        
        return crossings
    
    def dominant_constructions(self, fable):
        """
        Identify common word order patterns in the fable.
        
        Patterns:
        - Subject-Verb-Object ordering
        - Modifier positions (adjectives, adverbs)
        - Adposition patterns (prepositions/postpositions)
        
        Returns:
            Dict with patterns and frequencies
        """
        sentences = fable.get('sentences', [])
        language = fable.get('language', 'en')
        
        # Initialize results
        results = {
            'word_order_patterns': {
                'SVO': 0,  # Subject-Verb-Object
                'SOV': 0,  # Subject-Object-Verb
                'VSO': 0,  # Verb-Subject-Object
                'VOS': 0,  # Verb-Object-Subject
                'OVS': 0,  # Object-Verb-Subject
                'OSV': 0,  # Object-Subject-Verb
                'other': 0  # Incomplete or complex patterns
            },
            'adjective_positions': {
                'before_noun': 0,
                'after_noun': 0
            },
            'adposition_positions': {
                'preposition': 0,   # Before object (like English "in the house")
                'postposition': 0,  # After object (like Japanese "house in")
                'circumposition': 0 # Around object (like German "um...herum")
            },
            'language_expectations': {},
            'total_clauses_analyzed': 0,
            'total_adj_noun_pairs': 0,
            'total_adpositions': 0
        }
        
        # Language-specific expectations
        language_expectations = {
            'en': {
                'expected_word_order': 'SVO', 
                'expected_adj_position': 'before_noun',
                'expected_adposition': 'preposition'
            },
            'es': {
                'expected_word_order': 'SVO', 
                'expected_adj_position': 'after_noun',
                'expected_adposition': 'preposition'
            },
            'de': {
                'expected_word_order': 'SOV/SVO mixed', 
                'expected_adj_position': 'before_noun',
                'expected_adposition': 'preposition/circumposition'
            },
            'nl': {
                'expected_word_order': 'SOV/SVO mixed', 
                'expected_adj_position': 'before_noun',
                'expected_adposition': 'preposition/circumposition'
            },
            'grc': {
                'expected_word_order': 'flexible (SOV common)', 
                'expected_adj_position': 'flexible',
                'expected_adposition': 'preposition'
            }
        }
        
        if language in language_expectations:
            results['language_expectations'] = language_expectations[language]
        
        # Process each sentence
        for sentence in sentences:
            if 'dependencies' in sentence and 'tokens' in sentence and 'pos_tags' in sentence:
                deps = sentence['dependencies']
                tokens = sentence['tokens']
                pos_tags = sentence['pos_tags']
                
                # Map token IDs to their POS tags
                token_id_to_pos = {}
                for token, pos in zip(tokens, pos_tags):
                    if isinstance(token, dict) and 'id' in token:
                        token_id = token['id']
                        # Handle different POS tag formats
                        if isinstance(pos, tuple) and len(pos) >= 2:
                            token_id_to_pos[token_id] = pos[1]  # (token, POS) format
                        elif isinstance(pos, dict) and 'pos' in pos:
                            token_id_to_pos[token_id] = pos['pos']  # {pos: "TAG"} format
                        else:
                            token_id_to_pos[token_id] = str(pos)  # Direct POS value
                
                # Extract subject-verb-object relationships
                subjects = []
                objects = []
                verbs = []
                
                for dep in deps:
                    dep_type = dep.get('dep', '')
                    head_id = dep.get('head_id')
                    dep_id = dep.get('dependent_id')
                    
                    # Extract subject
                    if dep_type in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
                        subjects.append((dep_id, head_id))  # (subject_id, verb_id)
                    
                    # Extract object
                    elif dep_type in ['dobj', 'obj', 'iobj']:
                        objects.append((dep_id, head_id))  # (object_id, verb_id)
                    
                    # Track verbs
                    elif dep_id in token_id_to_pos and token_id_to_pos[dep_id] in ['VERB', 'AUX']:
                        verbs.append(dep_id)
                
                # Analyze clauses with subject, verb, and object
                for subj, verb_id in subjects:
                    for obj, obj_verb_id in objects:
                        # Only analyze if subject and object belong to the same verb
                        if verb_id == obj_verb_id:
                            # Get positions
                            subj_pos = next((i for i, t in enumerate(tokens) if t.get('id') == subj), -1)
                            verb_pos = next((i for i, t in enumerate(tokens) if t.get('id') == verb_id), -1)
                            obj_pos = next((i for i, t in enumerate(tokens) if t.get('id') == obj), -1)
                            
                            # Only continue if we found all positions
                            if subj_pos >= 0 and verb_pos >= 0 and obj_pos >= 0:
                                # Determine word order pattern
                                pattern = self._get_word_order(subj_pos, verb_pos, obj_pos)
                                if pattern in results['word_order_patterns']:
                                    results['word_order_patterns'][pattern] += 1
                                else:
                                    results['word_order_patterns']['other'] += 1
                                
                                results['total_clauses_analyzed'] += 1
                
                # Analyze adjective positions
                for dep in deps:
                    if dep.get('dep') == 'amod':  # Adjectival modifier
                        adj_id = dep.get('dependent_id')
                        noun_id = dep.get('head_id')
                        
                        # Get positions
                        adj_pos = next((i for i, t in enumerate(tokens) if t.get('id') == adj_id), -1)
                        noun_pos = next((i for i, t in enumerate(tokens) if t.get('id') == noun_id), -1)
                        
                        if adj_pos >= 0 and noun_pos >= 0:
                            if adj_pos < noun_pos:
                                results['adjective_positions']['before_noun'] += 1
                            else:
                                results['adjective_positions']['after_noun'] += 1
                                
                            results['total_adj_noun_pairs'] += 1
                
                # Analyze adposition patterns
                for dep in deps:
                    if dep.get('dep') in ['case', 'prep', 'mark']:  # Adpositions
                        adp_id = dep.get('dependent_id')
                        obj_id = dep.get('head_id')
                        
                        # Get positions
                        adp_pos = next((i for i, t in enumerate(tokens) if t.get('id') == adp_id), -1)
                        obj_pos = next((i for i, t in enumerate(tokens) if t.get('id') == obj_id), -1)
                        
                        if adp_pos >= 0 and obj_pos >= 0:
                            if adp_pos < obj_pos:
                                results['adposition_positions']['preposition'] += 1
                            else:
                                results['adposition_positions']['postposition'] += 1
                                
                            results['total_adpositions'] += 1
        
        # Calculate percentages
        if results['total_clauses_analyzed'] > 0:
            total = results['total_clauses_analyzed']
            results['word_order_percentages'] = {
                order: (count / total * 100) for order, count in results['word_order_patterns'].items()
            }
            
            # Find dominant pattern
            dominant = max(results['word_order_patterns'].items(), key=lambda x: x[1])
            results['dominant_word_order'] = dominant[0]
        
        if results['total_adj_noun_pairs'] > 0:
            total_adj = results['total_adj_noun_pairs']
            before_pct = (results['adjective_positions']['before_noun'] / total_adj * 100)
            after_pct = (results['adjective_positions']['after_noun'] / total_adj * 100)
            
            results['adjective_position_percentages'] = {
                'before_noun': before_pct,
                'after_noun': after_pct
            }
            
            # Determine dominant position
            if before_pct > after_pct:
                results['dominant_adj_position'] = 'before_noun'
            else:
                results['dominant_adj_position'] = 'after_noun'
        
        if results['total_adpositions'] > 0:
            total_adp = results['total_adpositions']
            results['adposition_percentages'] = {
                position: (count / total_adp * 100) 
                for position, count in results['adposition_positions'].items()
            }
            
            # Determine dominant type
            dominant = max(results['adposition_positions'].items(), key=lambda x: x[1])
            results['dominant_adposition'] = dominant[0]
        
        return results
    
    def _get_word_order(self, subj_pos, verb_pos, obj_pos):
        """
        Determine word order pattern from positions.
        
        Args:
            subj_pos: Position of subject
            verb_pos: Position of verb
            obj_pos: Position of object
            
        Returns:
            String representing word order pattern (SVO, SOV, etc.)
        """
        # Check for invalid positions
        if subj_pos < 0 or verb_pos < 0 or obj_pos < 0:
            return "other"
            
        # Check for unusually distant components (typically indicates parsing issues)
        max_distance = 30
        if (abs(subj_pos - verb_pos) > max_distance or 
            abs(verb_pos - obj_pos) > max_distance or
            abs(subj_pos - obj_pos) > max_distance):
            return "other"
        
        positions = [
            ('S', subj_pos),
            ('V', verb_pos),
            ('O', obj_pos)
        ]
        
        # Sort by position
        positions.sort(key=lambda x: x[1])
        
        # Build pattern string
        pattern = ''.join(pos[0] for pos in positions)
        
        return pattern


    def semantic_roles(self, fable):
        """
        Map semantic roles in the fable.
        
        Note: This is a simplified version that maps syntactic relations to
        semantic roles. For proper semantic role labeling, additional NLP 
        libraries would be needed.
        
        Returns:
            Dict with semantic role information
        """
        sentences = fable.get('sentences', [])
        language = fable.get('language', 'en')
        
        # Initialize results
        results = {
            'roles': {
                'Agent': [],       # Who performed the action
                'Patient': [],     # Who received the action
                'Recipient': [],   # Who received the object
                'Instrument': [],  # What was used for the action
                'Location': [],    # Where the action occurred
                'Time': [],        # When the action occurred
                'Manner': [],      # How the action was performed
                'Cause': [],       # Why the action occurred
                'Purpose': []      # For what purpose
            },
            'frequent_agents': [],
            'frequent_patients': [],
            'role_distribution': {},
            'total_roles': 0
        }
        
        # Simple mapping from dependency types to semantic roles
        # This is a simplification; proper semantic role labeling would require more context
        dep_to_role = {
            'nsubj': 'Agent',
            'agent': 'Agent',
            'dobj': 'Patient',
            'obj': 'Patient',
            'iobj': 'Recipient',
            'obl': 'Location',  # Could be location, time, or manner
            'nmod': 'Location',  # Could be various roles
            'advmod': 'Manner',
            'xcomp': 'Purpose',
            'advcl': 'Cause'
        }
        
        # Language-specific dependency mappings
        language_deps = {
            'en': {'nsubj': 'Agent', 'dobj': 'Patient'},
            'es': {'nsubj': 'Agent', 'obj': 'Patient'},
            'de': {'nsubj': 'Agent', 'obj': 'Patient', 'iobj': 'Recipient'},
            'nl': {'nsubj': 'Agent', 'obj': 'Patient', 'iobj': 'Recipient'},
            'grc': {'nsubj': 'Agent', 'obj': 'Patient'}
        }
        
        # Use language-specific mappings if available
        if language in language_deps:
            for dep, role in language_deps[language].items():
                dep_to_role[dep] = role
        
        # Process each sentence
        role_counts = Counter()
        agent_counts = Counter()
        patient_counts = Counter()
        
        for sentence in sentences:
            if 'dependencies' in sentence and 'tokens' in sentence:
                deps = sentence['dependencies']
                tokens = sentence['tokens']
                
                # Map token IDs to token text
                token_id_to_text = {}
                for token in tokens:
                    if isinstance(token, dict) and 'id' in token and 'text' in token:
                        token_id_to_text[token['id']] = token['text']
                    elif isinstance(token, (list, tuple)) and len(token) >= 2:
                        # Handle (id, text) tuple format
                        token_id_to_text[token[0]] = token[1]
                
                # Process each dependency
                for dep in deps:
                    dep_type = dep.get('dep', '')
                    dependent_id = dep.get('dependent_id')
                    head_id = dep.get('head_id')
                    
                    # Skip if we're missing information
                    if not dep_type or dependent_id not in token_id_to_text:
                        continue
                    
                    # Map to semantic role
                    if dep_type in dep_to_role:
                        role = dep_to_role[dep_type]
                        term = token_id_to_text[dependent_id]
                        
                        # Add to appropriate role list
                        if role in results['roles']:
                            results['roles'][role].append(term)
                        
                        # Track counts
                        role_counts[role] += 1
                        results['total_roles'] += 1
                        
                        # Track agents and patients specifically
                        if role == 'Agent':
                            agent_counts[term.lower()] += 1
                        elif role == 'Patient':
                            patient_counts[term.lower()] += 1
        
        # Calculate most frequent agents and patients
        results['frequent_agents'] = [
            {'term': term, 'count': count}
            for term, count in agent_counts.most_common(5)
        ]
        
        results['frequent_patients'] = [
            {'term': term, 'count': count}
            for term, count in patient_counts.most_common(5)
        ]
        
        # Calculate role distribution
        if results['total_roles'] > 0:
            results['role_distribution'] = {
                role: (count / results['total_roles'] * 100)
                for role, count in role_counts.items()
            }
        
        # Add language-specific notes
        language_notes = {
            'en': 'English typically shows Agent-Verb-Patient ordering',
            'es': 'Spanish allows pro-drop (omitting pronouns) which affects Agent detection',
            'de': 'German case marking makes semantic role identification more reliable',
            'nl': 'Dutch, like German, uses case marking but with different patterns',
            'grc': 'Ancient Greek used extensive case marking for semantic roles'
        }
        
        if language in language_notes:
            results['language_notes'] = language_notes[language]
        return results


    def save_analysis(self, fable_id, language, analysis_type, results):
        """
        Save analysis results to file.
        
        Args:
            fable_id: ID of the analyzed fable
            language: Language code
            analysis_type: Type of analysis (e.g., 'dependency', 'tree_shape')
            results: Analysis results to save
        """
        try:
            # Create directory if it doesn't exist
            output_dir = self.analysis_dir / 'syntax'
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create filename
            filename = f"{fable_id}_{language}_{analysis_type}.json"
            output_path = output_dir / filename
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Saved %s analysis for fable %s (%s) to %s",
                            analysis_type, fable_id, language, output_path)
                            
        except FileNotFoundError as e:
            self.logger.error("File path error when saving analysis: %s", e)
        except PermissionError as e:
            self.logger.error("Permission error when saving analysis: %s", e)
        except json.JSONDecodeError as e:
            self.logger.error("JSON encoding error when saving analysis: %s", e)
        except Exception as e:
            # Keep one general exception at the end as a fallback
            self.logger.error("Unexpected error when saving analysis: %s (%s)", 
                            e, type(e).__name__)
    
    def compare_fables(self, fables_by_id, analysis_type):
        """
        Compare syntax across different language versions of the same fable.
        
        Args:
            fables_by_id: Dict mapping fable IDs to language-specific fables
            analysis_type: Type of analysis to compare
            
        Returns:
            Dict with comparison results
        """
        comparison = {}
        
        for fable_id, lang_fables in fables_by_id.items():
            fable_comparison = {
                'languages': list(lang_fables.keys()),
                'results': {}
            }
            
            for lang, fable in lang_fables.items():
                # Run the requested analysis
                if analysis_type == 'dependency_frequencies':
                    results = self.dependency_frequencies(fable)
                elif analysis_type == 'dependency_distances':
                    results = self.dependency_distances(fable)
                elif analysis_type == 'tree_shapes':
                    results = self.tree_shapes(fable)
                elif analysis_type == 'dominant_constructions':
                    results = self.dominant_constructions(fable)
                elif analysis_type == 'semantic_roles':
                    results = self.semantic_roles(fable)
                else:
                    results = {'error': f'Unknown analysis type: {analysis_type}'}
                
                fable_comparison['results'][lang] = results
            
            comparison[fable_id] = fable_comparison
        
        return comparison