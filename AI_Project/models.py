from transformers import pipeline
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag


device = 0 if torch.cuda.is_available() else -1

sent_pipeline = pipeline(
    "sentiment-analysis",
    model="tabularisai/multilingual-sentiment-analysis",
    device=device
)

sum_pipeline = pipeline(
    "summarization",
    model="t5-small",
    device=device
)

stopwords = set(stopwords.words("english"))

def sentiment_analysis(text: str):
    result = sent_pipeline(text)[0]

    label = result["label"].lower()

    if label == "positive":
        return "positive"
    elif label == "negative":
        return "negative"
    else:
        return "neutral"
    

def extract_keyword(text : str, top_k = 5):
    tokens =  [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stopwords]
    filtered = [w for w in tokens]

    tagged = pos_tag(filtered)
    keywords = [w for w, pos in tagged if pos.startswith("NN") or pos.startswith("JJ")]

    return [k.lower() for k in keywords][:top_k]
