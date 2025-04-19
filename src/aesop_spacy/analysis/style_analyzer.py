# style_analyzer.py
"""
Compares stylistic Elements across languages by analyzing:
* Sentence Complexity: Measure how intricate sentences are. 
* Lexical Richness: Assess vocabulary diversity.
* Identify persuasive or stylistic techniques.
"""

class StyleAnalyzer:
    """
    Docstring
    """
    def __init__(self, analysis_dir):
        pass
    def sentence_complexity(self, fable):
        """Analyze sentence complexity metrics.
           #metrics:
           #average sentence length -> no of tokens per sentence
           #average clauses per sentence -> indicates subordination or complexity
        """
        # Extract sentences
        sentences = fable.get('sentences', [])
        sentence_count = len(sentences)

        if sentence_count == 0:
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'avg_clauses_per_sentence': 0
            }

        # Calculate token counts
        token_counts = []
        for sentence in sentences:
            # Handle potential missing text
            text = sentence.get('text', '')
            # Split on whitespace and filter out empty strings
            tokens = [t for t in text.split() if t]
            token_counts.append(len(tokens))

        # Calculate average sentence length 
        avg_sentence_length = sum(token_counts) / sentence_count

        # Analyze clauses per sentence
        clause_counts = []

        for sentence in sentences:
            # Start with the main clause (ROOT)
            clause_count = 1
            text = sentence.get('text', '').lower()
            # Check for common subordinating conjunctions in English
            subordinate_words = ['because', 'if', 'when', 'while', 'that', 'than', 
                                'as', 'since', 'although', 'though', 'to']
         
            for word in subordinate_words:
                if f' {word} ' in f' {text} ':  # Spaces ensure whole word matches
                    clause_count += 1
         
            # Extract tokens if available for more precise analysis
            sentence_tokens = []
            if 'tokens' in sentence:
                sentence_tokens = sentence['tokens']
            elif 'pos_tags' in fable and 'text' in sentence:
                # Fallback if tokens aren't directly attached to sentences
                start, end = sentence.get('start', 0), sentence.get('end', 0)
                sentence_tokens = fable.get('tokens', [])[start:end]
        
            # Count subordinating conjunctions as potential clause markers
            for token in sentence_tokens: 
                # Handle different data structures
                if isinstance(token, list) and len(token) >= 2:
                    # [token_text, token_info] format
                    pos = token[1] if isinstance(token[1], str) else token[1].get('pos', '')
                elif isinstance(token, dict):
                    # {pos: ...} format
                    pos = token.get('pos', '')
                else:
                    # Fallback - try to use pos_tags if available
                    pos = ''
            
                # Count subordinating conjunctions
                if pos in ['SCONJ']:
                    clause_count += 1
        
            clause_counts.append(clause_count)
    
        # Calculate average clauses per sentence
        avg_clauses = sum(clause_counts) / sentence_count
    
        return {
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length, 
            'avg_clauses_per_sentence': avg_clauses
        }

    def lexical_richness(self,fable):
        """Analyze sentence complexity metrics"""
        pass
    def rhetorical_devices(self,fable):
        """Detect rhetorical devices like repetition and alliteration."""
        pass

    def _max_depth(self,token):
        """Helper: Calculate max depth of a dependency tree."""
        if not list(token.children):
            return 1
        return 1 + max(self._max_depth(child) for child in token.children)
    
