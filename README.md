🚧 Status: Code is being refactored and bugs fixed

<div align="center">
  <h1>Multilingual Fables Analyzer</h1>
  <p><em>Analyzing Aesop's fables across languages using advanced NLP techniques</em></p>
  <img src="./aesop_cover.png" alt="Aesop's Fables Illustration" width="500">
</div>

## 📖 What is this?

This project analyzes Aesop's fables in multiple languages (English, German, Dutch, Spanish, and Ancient Greek) to discover interesting patterns and comparisons. It uses natural language processing (NLP) to understand how stories are told differently across languages.

* It's like having a super-powered reading assistant that can analyze the same story in different languages and tell you how they compare!

## ✨ Features

- **Read and process fables** in multiple languages
- **Compare how stories differ** when told in different languages
- **Analyze writing styles** to see what makes each language unique
- **Explore morals and themes** across cultural tellings
- **Visualize the results** with colorful, easy-to-understand graphs

## 🔬 Technologies & Linguistic Approaches

This project combines several computational linguistics techniques:

- **Multilingual Processing**: Working with 5 languages including English, German, Dutch, Spanish and Ancient Greek
- **Part-of-Speech Analysis**: Examining how different grammatical structures work across languages
- **Dependency Parsing**: Understanding the relationships between words in sentences
- **Named Entity Recognition**: Identifying characters, locations, and other entities
- **Semantic Similarity**: Comparing meaning preservation across translations
- **Sentiment Analysis**: Analyzing emotional tone in different language versions
- **Stylometric Analysis**: Measuring writing style features like sentence complexity
- **Topic Modeling**: Discovering themes across multilingual corpus
- **Vector Embeddings**: Representing words and concepts in mathematical space

Industry-standard NLP libraries are used:

- **spaCy**: For core linguistic analysis
- **NLTK**: For supplementary text processing
- **Stanza**: For languages with specific requirements
- **Hugging Face Transformers**: For advanced multilingual models



## 🚀 Getting Started

### For everyone

If you're just curious about the project:

1. Browse the `notebooks` folder to check some early explorations and playing around nlp
2. Check out the original fables in the `data/raw/fables` directory
3. Run the main.py and see the analysis and figures output 

### For developers 

#### Clone and setup
git clone https://github.com/yourusername/RosetaFables.git
cd RosetaFables
poetry install

#### Activate environment and run
poetry shell
python main.py

#### Run tests
poetry run pytest

#### Run specific analysis
poetry run python -m src.aesop_spacy.analysis.style_analyzer


