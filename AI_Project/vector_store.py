from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import mlflow

class VectorStore:
    def __init__(self, store_folder="faiss_index"):
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )

        self.faiss_store = Path(store_folder)
        self.faiss_store.mkdir(exist_ok=True)
        self.faiss_index = None
        self.text_store = []

        self.version_file = self.faiss_store / "version.txt"
        if self.version_file.exists():
            self.version = int(self.version_file.read_text()) + 1
        else:
            self.version = 1
        self.version_file.write_text(str(self.version))

        self.faiss_index_path = self.faiss_store / f"index_v{self.version}"

        if (self.faiss_store / f"index_v{self.version-1}").exists():
            self.faiss_index = FAISS.load_local(
                str(self.faiss_store / f"index_v{self.version-1}"),
                self.embed_model
            )


    def add_text(self, text: str):
        """Add a text string to FAISS index."""
        doc = Document(page_content=text)

        if self.faiss_index is None:
            self.faiss_index = FAISS.from_documents(
                documents=[doc],
                embedding=self.embed_model
            )
        else:
            self.faiss_index.add_documents([doc])

        self.faiss_index.save_local(str(self.faiss_index_path))
        self.text_store.append(text)

    def search(self, query: str, top_k: int = 3):
        """Search similar text."""
        query_embedding = self.embed_model.embed_query(query)
        results = self.faiss_index.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            k=top_k
        )

        output = []
        for doc, score in results:
            output.append({
                "text": doc.page_content,
                "score": float(score)  
            })
        return output


vector_db = VectorStore()
