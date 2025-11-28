# 🧠 NLP Analyzer API  
A production-ready NLP service built using **FastAPI**, **HuggingFace Transformers**, **LangChain FAISS Vector Store**, and **MLflow** for experiment tracking and model comparison.

This project provides:
- ✔ Sentiment Analysis  
- ✔ Text Summarization  
- ✔ Semantic Search using FAISS  
- ✔ Automatic Vector Store Versioning  
- ✔ MLflow-based Model Performance Tracking  
- ✔ Dockerized Deployment  
- ✔ (Optional) CI/CD Pipeline Support  

---

## 🚀 Features

### **1. Sentiment Analysis**
Uses:
- `tabularisai/multilingual-sentiment-analysis`

Logged with:
- Model name  
- Latency  
- Input size  
- Output  

MLflow allows comparing future models simply by replacing the model path.

---

### **2. Text Summarization**
Uses:
- `t5-small`

MLflow logs:
- Model name  
- Processing time  
- Full input/output artifacts  
- Summary quality testing over time  

You can easily replace the model with:
- `facebook/bart-large-cnn`
- `google/pegasus-xsum`
- `Falcon-instruct`, etc.

---

### **3. Semantic Search (FAISS + LangChain)**
- Embedding Model: **all-mpnet-base-v2**
- Every FAISS index is saved with **versioning**
- Search uses vector similarity.
- New texts are automatically appended and indexed.

---

### **4. Automatic Model & Index Versioning**
Each time your app starts:
- `version.txt` increments
- A new FAISS index folder is created
- MLflow logs the version used

This makes the project **fully auditable & MLOps compliant**.

---

### **5. MLflow Tracking**
Track:
- Sentiment Model Performance  
- Summarization Model Performance  
- FAISS Index Version  
- Latency Metrics  
- Input Length  
- Output Artifacts  


## **Setup Instructions**

### 1. Clone the repository
    ```bash
    git clone https://github.com/Abinash200501/NLP_Task.git
    cd AI_Project

### 2.  Install dependicies

    ```bash
    python -m venv myenv
    source myenv/bin/activate       # Linux/macOS
    myenv\Scripts\activate          # Windows
    pip install --upgrade pip
    pip install -r requirements.txt

### 3.Run with FastAPI

    ```bash
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    Swagger UI is available at "http://localhost:8000/docs"

### 4. MLflow Tracking (MLOps)
    We use MLflow to track model behavior & store experiment logs.

    ```bash
    mlflow ui --port 5000

    Open in browser at "http://127.0.0.1:5000




