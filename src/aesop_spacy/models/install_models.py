import subprocess

models = [
    "de_dep_news_trf",
    "nl_core_news_lg",
    "xx_sent_ud_sm"
]

for model in models:
    subprocess.run(["python", "-m", "spacy", "download", model], check=True)