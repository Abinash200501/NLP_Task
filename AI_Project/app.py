from fastapi import FastAPI
from schemas import Analyze, Summary, SemanticSearchRequest
from models import sentiment_analysis, extract_keyword, sum_pipeline
from vector_store import vector_db
from pathlib import Path

app = FastAPI()

@app.get("/home")
def root():
    return {"message": "API is Running"}

@app.post("/analyze")
def analyze(req : Analyze):

    text = req.text
    sentiment = sentiment_analysis(text)
    keywords = extract_keyword(text=text, top_k=5)

    data = Path("data")

    data.mkdir(exist_ok=True, parents=True)

    data_file = data / "text.txt"
    with open(data_file, "a") as f:
        f.write(text + "\n")

    vector_db.add_text(text=text)

    return {
        "sentiment": sentiment,
        "keywords": keywords
    }


@app.post("/summarize")
def summarize(req : Summary):
    text = req.summary

    if len(text.split()) == 0:
        raise Exception 
    
    if len(text.split()) < 5:
        return {"Summary": "Text is too short, please provide valid input"}
    
    summary_result = sum_pipeline(text, max_length = 100, min_length = 50)
    summary_text = summary_result[0]["summary_text"]

    return {"summary": summary_text}


@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    query_text = req.text
    results = vector_db.search(query_text)
    return {"results": results}