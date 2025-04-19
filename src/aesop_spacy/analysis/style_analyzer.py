# style_analyzer.py
"""
Compares stylistic Elements across languages by analyzing:
* Sentence Complexity: Measure how intricate sentences are. 
* Lexical Richness: Assess vocabulary diversity.
* Identify persuasive or stylistic techniques.
"""

from collections import Counter
class StyleAnalyzer:
    def __init__(self, analysis_dir):
        pass
    def sentence_complexity(self, fable):
        """Analyze sentence complexity metrics."""
        #metrics:
        # average sentence length -> no of tokens per sentence
        # Extract sentences
        sentences = fable.get('sentences', [])
        sentence_count = len(sentences)
        if sentence_count == 0:
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'avg_clauses_per_sentence':0
            }
        
        token_counts = []
        for sentence in sentences:
            # Handle potential missing text
            text = sentence.get('text', '')
            # Split on whitespace and filter out of empty strings
            tokens = [t for t in text.split() if t]
            token_counts.append(len(tokens))

          # Calculate average sentence length 
        avg_sentence_length = sum(token_counts) / sentence_count
        # -----------
        # average clauses per sentence -> indicates subordination or complexity
        clause_counts = []

        for sentence in sentences:
            # checking if we have token dependency information
            tokens = sentence.get('tokens', [])
            clause_count = 0                

            # Count tokens that represent clause heads
            for token in tokens:
                dep = token.get('dep', '')
                if dep in ['ROOT', 'ccomp', 'xcomp', 'advcl', 'relcl']:
                    clause_count += 1

            # at least one clause per valid sentence
            clause_count = max(1, clause_count) if tokens else 0
            clause_counts.append(clause_count)

            # calculate average
            avg_clauses = sum(clause_counts) / sentence_count if sentence_count > 0 else 0
        
        # --------
        # average dependency tree depth -> shows syntatic nesting

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
    
