import spacy
nlp = spacy.load("nl_core_news_lg")
with open("aesop_extract_nlp_nl_de.txt", encoding="utf8" ,errors='ignore') as f:
    text = f.read()
print(text)
