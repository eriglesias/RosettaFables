# style_analyzer.py
"""
Compares stylistic Elements across languages by analyzing:
* Sentence Complexity: Measure how intricate sentences are. 
* Lexical Richness: Assess vocabulary diversity.
* Identify persuasive or stylistic techniques.
"""
import math
class StyleAnalyzer:
    """
    Docstring
    """
    def __init__(self, analysis_dir):
        self.analysis_dir = analysis_dir


    def sentence_complexity(self, fable):
        """
        Analyze sentence complexity metrics with multilingual support.
        Measures:
        - Average sentence length (in tokens)
        - Average clauses per sentence
        - Average dependency tree depth
        """
        # Extract sentences
        sentences = fable.get('sentences', [])
        sentence_count = len(sentences)

        if sentence_count == 0:
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'avg_clauses_per_sentence': 0,
                'avg_dependency_depth': 0
            }

        # fetching the language coe from the fable, defaulting to 'en' if not available
        language = fable.get('language', 'en')
        # calculating token counts
        token_counts = []
        for sentence in sentences:
            text = sentence.get('text', '')
            tokens = [t for t in text.split() if t]
            token_counts.append(len(tokens))
        
        avg_sentence_length = sum(token_counts) / sentence_count
        
        # 2. Analyze clauses per sentence
        clause_counts = []
        
        # Language-specific subordinating conjunctions
        subordinators = {
            'en': ['because', 'if', 'when', 'while', 'that', 'than', 'as', 'since', 'although', 'though'],
            'es': ['que', 'cuando', 'si', 'aunque', 'porque', 'como', 'mientras'],
            'de': ['dass', 'wenn', 'als', 'ob', 'weil', 'damit', 'obwohl'],
            'nl': ['dat', 'als', 'toen', 'omdat', 'hoewel', 'terwijl'],
            'grc': ['ὅτι', 'ἵνα', 'ὡς', 'εἰ', 'ἐάν', 'ὅπως']  # Approximate for Ancient Greek
        }
        
        lang_subordinators = subordinators.get(language, subordinators['en'])
        
        for sentence in sentences:
            # Start with the main clause (ROOT)
            clause_count = 1
            
            # Look for subordinating conjunctions in the text
            text = sentence.get('text', '').lower()
            words = text.split()
            for word in words:
                # Remove punctuation for matching
                word = word.strip('.,;:!?"\'')
                if word in lang_subordinators:
                    clause_count += 1
            
            # If we have POS tags, use those for more precision
            if 'pos_tags' in sentence:
                for tag in sentence['pos_tags']:
                    if isinstance(tag, list) and len(tag) >= 2 and tag[1] == 'SCONJ':
                        clause_count += 1
                    elif isinstance(tag, dict) and tag.get('pos') == 'SCONJ':
                        clause_count += 1
            
            clause_counts.append(clause_count)
        
        avg_clauses = sum(clause_counts) / sentence_count
        
        # 3. Calculate dependency tree depth
        depth_values = []
        
        for sentence in sentences:
            # If we have the full dependency tree with root information
            if 'root' in sentence and isinstance(sentence['root'], dict):
                root_token = sentence['root']
                if 'tokens' in sentence:
                    # Build a graph representation of the dependency tree
                    token_map = {i: token for i, token in enumerate(sentence['tokens'])}
                    children_map = {}
                    for i, token in token_map.items():
                        head = token.get('head', -1)
                        if head not in children_map:
                            children_map[head] = []
                        children_map[head].append(i)
                    
                    # Calculate max depth
                    root_id = root_token.get('id', 0)
                    max_depth = self._max_depth_helper(root_id, children_map)
                    depth_values.append(max_depth)
                else:
                    # Fallback to an estimation
                    tokens = sentence.get('text', '').split()
                    estimated_depth = 1 + int(math.log(len(tokens) + 1))
                    depth_values.append(estimated_depth)
            else:
                # If no dependency info, estimate based on sentence length
                
                tokens = sentence.get('text', '').split()
                # tree depth often correlates logarithmically with sentence length
                estimated_depth = 1 + int(math.log(len(tokens) + 1))
                depth_values.append(estimated_depth)
        
        avg_depth = sum(depth_values) / sentence_count if depth_values else 0
        
        return {
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_clauses_per_sentence': avg_clauses,
            'avg_dependency_depth': avg_depth
        }

    def _max_depth_helper(self, token_id, children_map):
        """Helper method to calculate the maximum depth of the dependency tree."""
        if token_id not in children_map or not children_map[token_id]:
            return 1
        child_depths = [self._max_depth_helper(child_id, children_map) for child_id in children_map[token_id]]
        return 1 + max(child_depths)

    def lexical_richness(self, fable):
        """
        Analyze lexical richness with multiple metrics.
        
        Metrics:
        - Type-Token Ratio (TTR): Ratio of unique words to total words
        - Standardized TTR (STTR): Average TTR for segments of fixed length
        - Moving Average TTR (MATTR): TTR computed over a sliding window
        - Hapax Ratio: Proportion of words occurring exactly once
        - Vocabulary Growth: Curve showing vocabulary size growth over text
        """
        # Extract text (prioritize body, fallback to sentences)
        full_text = fable.get('body', '')
        if not full_text and 'sentences' in fable:
            sentences = fable.get('sentences', [])
            full_text = ' '.join(sentence.get('text', '') for sentence in sentences)
        
        # Get language for language-specific processing
        language = fable.get('language', 'en')
        
        # Tokenize appropriately based on available data
        tokens = [t.lower() for t in full_text.split() if t.strip()]
        tokens = [t for t in tokens if t and len(t) > 1 and not all(c in '.,;:!?"\'()[]{}' for c in t)]
        
        # If lemmas are available, use them for better normalization
        if 'lemmas' in fable and isinstance(fable['lemmas'], list):
            # Lemmas normalize word forms (running -> run)
            tokens = [lemma[0].lower() for lemma in fable['lemmas'] if isinstance(lemma, list) and lemma]
        elif 'tokens' in fable and isinstance(fable['tokens'], list):
            tokens = [token[0].lower() for token in fable['tokens'] if isinstance(token, list) and token]
        else:
            # Simple tokenization as fallback (not ideal but works)
            tokens = [t.lower() for t in full_text.split() if t.strip()]
        
        # Filter out punctuation and very short tokens (if appropriate for the language)
        tokens = [t for t in tokens if t and len(t) > 1 and not all(c in '.,;:!?"\'()[]{}' for c in t)]
        
        # Calculate basic counts
        token_count = len(tokens)
        types = set(tokens)
        type_count = len(types)
        
        # Handle empty text
        if token_count == 0:
            return {
                'token_count': 0,
                'type_count': 0,
                'ttr': 0,
                'sttr': 0,
                'mattr': 0,
                'hapax_count': 0,
                'hapax_ratio': 0,
                'vocab_growth': []
            }
        
        # 1. Basic Type-Token Ratio (TTR)
        ttr = type_count / token_count
        
        # 2. Standardized TTR (STTR) - calculated for segments of fixed size
        segment_size = min(100, token_count // 2) if token_count > 50 else token_count
        
        # Only calculate STTR if we have enough tokens
        if token_count >= segment_size:
            segments = [tokens[i:i+segment_size] for i in range(0, token_count, segment_size)]
            segment_ttrs = []
            
            for segment in segments:
                if len(segment) >= segment_size * 0.9:  # Only use mostly complete segments
                    segment_types = set(segment)
                    segment_ttrs.append(len(segment_types) / len(segment))
                    
            sttr = sum(segment_ttrs) / len(segment_ttrs) if segment_ttrs else ttr
        else:
            sttr = ttr  # Fall back to regular TTR for short texts
        
        # 3. Moving Average TTR (MATTR)
        window_size = min(50, token_count) if token_count > 20 else token_count
        mattr_values = []
        
        # Only calculate MATTR if we have enough tokens
        if token_count >= window_size:
            for i in range(token_count - window_size + 1):
                window = tokens[i:i+window_size]
                window_types = set(window)
                mattr_values.append(len(window_types) / window_size)
                
            mattr = sum(mattr_values) / len(mattr_values) if mattr_values else ttr
        else:
            mattr = ttr  # Fall back to regular TTR for short texts
        
        # 4. Hapax legomena (words appearing exactly once)
        word_counts = {}
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
            
        hapax_words = [word for word, count in word_counts.items() if count == 1]
        hapax_count = len(hapax_words)
        hapax_ratio = hapax_count / token_count
        
        # 5. Vocabulary growth curve
        # Sample at regular intervals through the text
        num_points = min(20, token_count)  # 20 points or fewer for short texts
        vocab_growth = []
        seen_words = set()
        
        sample_intervals = [i * token_count // num_points for i in range(1, num_points + 1)]
        for position in sample_intervals:
            # Words seen up to this position
            seen_words.update(tokens[:position])
            vocab_growth.append({
                'position': position,
                'unique_words': len(seen_words)
            })
        
        return {
            'token_count': token_count,
            'type_count': type_count,
            'ttr': ttr,
            'sttr': sttr,
            'mattr': mattr,
            'hapax_count': hapax_count, 
            'hapax_ratio': hapax_ratio,
            'vocab_growth': vocab_growth
        }

    def rhetorical_devices(self, fable):
        """
        Detect rhetorical devices like repetition and alliteration.
        
        Devices:
        - Repetition: Repeated words/phrases/structures
        - Alliteration: Repeated initial sounds
        - Parallelism: Similar syntactic structures
        - Metaphors: Figurative language (estimated through patterns)
        """
        # Extract text and language
        language = fable.get('language', 'en')
        
        # Get sentences for analysis
        sentences = fable.get('sentences', [])
        if not sentences:
            return {'error': 'No sentence data available for rhetorical analysis'}
        
        # Get tokens and lemmas (if available)
        tokens = []
        lemmas = []
        
        # Extract tokens and lemmas from each sentence
        for sentence in sentences:
            text = sentence.get('text', '')
            if 'tokens' in sentence:
                sentence_tokens = sentence['tokens']
            else:
                # Simple fallback tokenization
                sentence_tokens = [(t, '') for t in text.split()]
                
            tokens.append(sentence_tokens)
            
            # Extract lemmas if available
            if 'lemmas' in sentence:
                sentence_lemmas = sentence['lemmas']
                lemmas.append(sentence_lemmas)
        
        # Results container
        results = {
            'repetition': [],
            'alliteration': [],
            'parallelism': [],
            'possible_metaphors': []
        }
        
        # 1. Detect repetition (repeated lemmas)
        all_lemmas = []
        for sentence_lemmas in lemmas:
            all_lemmas.extend(sentence_lemmas)
        
        # Count lemma frequencies
        lemma_counts = {}
        for lemma in all_lemmas:
            if isinstance(lemma, str):
                lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1
        
        # Find significant repetitions (excluding common words)
        stopwords = {
            'en': ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for'],
            'es': ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'pero', 'en', 'a', 'de', 'por'],
            'de': ['der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'auf', 'bei', 'zu', 'von', 'für'],
            'nl': ['de', 'het', 'een', 'en', 'of', 'maar', 'in', 'op', 'bij', 'tot', 'van', 'voor'],
            'grc': ['ὁ', 'ἡ', 'τό', 'καί', 'ἤ', 'ἀλλά', 'ἐν', 'ἐπί', 'παρά', 'πρός', 'ἀπό', 'διά']
        }
        
        lang_stopwords = stopwords.get(language, stopwords['en'])
        
        # Find and add significant repetitions
        for lemma, count in lemma_counts.items():
            if count > 2 and lemma.lower() not in lang_stopwords:
                results['repetition'].append({
                    'lemma': lemma,
                    'count': count
                })
        
        # Sort by count (descending)
        results['repetition'] = sorted(results['repetition'], key=lambda x: x['count'], reverse=True)
        
        # 2. Detect alliteration
        for i, sentence_tokens in enumerate(tokens):
            # Extract the first letter of each content word
            first_letters = []
            for j, token in enumerate(sentence_tokens):
                # Skip punctuation and short tokens
                token_text = token[0] if isinstance(token, (list, tuple)) else str(token)
                if len(token_text) > 1 and token_text[0].isalpha():
                    first_letters.append(token_text[0].lower())
            
            # Check for alliteration (3+ words starting with same letter)
            letter_counts = {}
            for letter in first_letters:
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
            
            for letter, count in letter_counts.items():
                if count >= 3:
                    sentence_text = sentences[i].get('text', '')
                    results['alliteration'].append({
                        'letter': letter,
                        'count': count,
                        'sentence': sentence_text
                    })
        
        # 3. Detect parallelism (sentences with similar structure)
        # This is a simplified approach - detecting sentences with similar length and POS patterns
        for i, sentence1 in enumerate(sentences[:-1]):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                # Get POS tags if available
                pos_pattern1 = self._extract_pos_pattern(sentence1)
                pos_pattern2 = self._extract_pos_pattern(sentence2)
                
                # Check for structural similarity
                if pos_pattern1 and pos_pattern2:
                    # Calculate similarity based on POS pattern
                    similarity = self._pattern_similarity(pos_pattern1, pos_pattern2)
                    
                    if similarity > 0.7:  # High similarity threshold
                        results['parallelism'].append({
                            'sentence1': sentence1.get('text', ''),
                            'sentence2': sentence2.get('text', ''),
                            'similarity': similarity
                        })
        
        # 4. Detect potential metaphors
        # Look for specific patterns that often indicate metaphors
        metaphor_patterns = {
            'en': [('is', 'a'), ('like', 'a'), ('as', 'as')],
            'es': [('es', 'un'), ('como', 'un'), ('parece')],
            'de': [('ist', 'ein'), ('wie', 'ein'), ('als', 'ob')],
            'nl': [('is', 'een'), ('als', 'een'), ('lijkt', 'op')],
            'grc': [('ἐστί'), ('ὥσπερ'), ('οἷον')]
        }
        
        lang_patterns = metaphor_patterns.get(language, metaphor_patterns['en'])
        
        for i, sentence in enumerate(sentences):
            text = sentence.get('text', '').lower()
            
            # Check for metaphoric patterns
            for pattern in lang_patterns:
                if isinstance(pattern, tuple):
                    if all(word in text for word in pattern):
                        results['possible_metaphors'].append({
                            'sentence': sentence.get('text', ''),
                            'pattern': ' + '.join(pattern)
                        })
                else:
                    if pattern in text:
                        results['possible_metaphors'].append({
                            'sentence': sentence.get('text', ''),
                            'pattern': pattern
                        })
        
        return results

    def _extract_pos_pattern(self, sentence):
        """Extract POS tag pattern from a sentence."""
        if 'pos_tags' in sentence:
            # Return just the POS tags
            return [pos for _, pos in sentence['pos_tags']]
        return []

    def _pattern_similarity(self, pattern1, pattern2):
        """Calculate similarity between two POS patterns."""
        # Simple approach: calculate longest common subsequence
        m, n = len(pattern1), len(pattern2)
        
        # Skip if patterns are vastly different in length
        if m == 0 or n == 0:
            return 0
        
        if abs(m - n) / max(m, n) > 0.3:  # More than 30% different in length
            return 0
        
        # Create a matrix to store lengths of LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Build the dp table
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif pattern1[i-1] == pattern2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Length of LCS
        lcs_length = dp[m][n]
        
        # Normalize by average length
        return 2 * lcs_length / (m + n)