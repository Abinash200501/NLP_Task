# AI NLP API

A FastAPI-based NLP project that provides **Text Sentiment Analysis**, **Keyword Extraction**, **Summarization**, and **Semantic Search** using **Transformers** and **FAISS embeddings**. The project is containerized with **Docker** and includes **automatic model versioning**.

---

## **Features**

1. **Text Sentiment & Keyword Analysis**
   - Endpoint: `/analyze`
   - Returns sentiment (`positive`, `negative`, `neutral`) and top 5 keywords from text.

2. **Text Summarization**
   - Endpoint: `/summarize`
   - Summarizes input text with Transformers.

3. **Semantic Search**
   - Endpoint: `/semantic-search`
   - Stores embeddings using **FAISS** and returns top similar texts based on cosine similarity.

4. **Model Versioning**
   - Text embeddings and FAISS index are versioned automatically for reproducibility.

---

## **Folder Structure**

AI_Project/
│
├─ app.py # FastAPI app
├─ models.py # NLP models for sentiment, summarization, keywords
├─ vector_store.py # FAISS-based VectorStore with versioning
├─ schemas.py # Pydantic schemas
├─ requirements.txt # Python dependencies
├─ Dockerfile # Docker setup
├─ README.md # Project documentation

## **Setup Instructions**

### 1. Clone the repository
    ```bash
    git clone <repo_url>
    cd AI_Project

### 2.  Install dependicies
python -m venv myenv
source myenv/bin/activate       # Linux/macOS
myenv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt

### 3.Run with FastAPI

uvicorn app:app --reload --host 0.0.0.0 --port 8000
Swagger UI is available at "http://localhost:8000/docs"



