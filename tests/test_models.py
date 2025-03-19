import spacy
import logging

def test_models():
    """Test loading of spaCy models."""
    models = [
        "xx_sent_ud_sm",   # Multilingual
        "nl_core_news_lg", # Dutch
        "de_dep_news_trf", # German
        "en_core_web_lg",  # English
        "es_core_news_md"  # Spanish
    ]

    results = {}

    for model_name in models:
        try:
            nlp = spacy.load(model_name)
            print(f"✅ Successfully loaded {model_name}")
            # Test with a simple sentence
            test_text = "This is a test sentence."
            doc = nlp(test_text)
            print(f"  - Tokens: {[token.text for token in doc]}")
            print(f"  - POS tags: {[(token.text, token.pos_) for token in doc]}")
            results[model_name] = True
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            results[model_name] = False

    # Summary
    print("\nModel loading summary:")
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model}")

if __name__ == "__main__":
    test_models()