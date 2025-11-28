from fastapi import FastAPI
from schemas import Analyze, Summary, SemanticSearchRequest
from models import sentiment_analysis, extract_keyword, sum_pipeline, sent_pipeline
from vector_store import vector_db
from pathlib import Path
import mlflow
import time

app = FastAPI()

mlflow.set_experiment("nlp-task")


@app.get("/home")
def root():
    return {"message": "API is Running"}


@app.post("/analyze")
def analyze(req: Analyze):

    text = req.text

    start = time.time()
    sentiment = sentiment_analysis(text)
    keywords = extract_keyword(text=text, top_k=5)
    inference_time = time.time() - start

    data = Path("data")
    data.mkdir(exist_ok=True, parents=True)

    data_file = data / "text.txt"
    with open(data_file, "a") as f:
        f.write(text + "\n")

    vector_db.add_text(text=text)

    model_name = sent_pipeline.model.name_or_path

    with mlflow.start_run(run_name=f"analyze-text-{model_name}"):

        mlflow.log_param("model", model_name)
        mlflow.log_param("faiss_version", vector_db.version)
        mlflow.log_param("input_length", len(text))
        
        mlflow.log_metric("inference_time_seconds", inference_time)

        with open("analyze_output.txt", "a") as f:
            f.write("INPUT:\n")
            f.write(text)
            f.write("\n\nSENTIMENT:\n")
            f.write(str(sentiment))
            f.write("\n\nKEYWORDS:\n")
            f.write(str(keywords))

        mlflow.log_artifact("analyze_output.txt")

    return {
        "sentiment": sentiment,
        "keywords": keywords
    }


@app.post("/summarize")
def summarize(req: Summary):

    text = req.summary

    if len(text.split()) == 0:
        raise Exception
    
    if len(text.split()) < 5:
        return {"Summary": "Text is too short, please provide valid input"}

    start = time.time()
    summary_result = sum_pipeline(text, max_length=100, min_length=50)
    summary_text = summary_result[0]["summary_text"]
    latency = time.time() - start

    model_name = sum_pipeline.model.name_or_path

    with mlflow.start_run(run_name=f"summarizer-{model_name}"):

        mlflow.log_param("model", model_name)
        mlflow.log_param("input_length", len(text))
        mlflow.log_metric("latency_seconds", latency)

        with open("summary_output.txt", "a") as f:
            f.write("INPUT:\n")
            f.write(text)
            f.write("\n\nOUTPUT:\n")
            f.write(summary_text)

        mlflow.log_artifact("summary_output.txt")

    return {"summary": summary_text}


@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):

    text = req.text

    start = time.time()
    results = vector_db.search(text)
    latency = time.time() - start

    with mlflow.start_run(run_name="semantic-search"):

        mlflow.log_param("input_length", len(text))
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("results_found", len(results))

    return {"results": results}
