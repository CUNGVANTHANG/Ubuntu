import os
import json # Thư viện dọc file JSON
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Embedding:
    # Khởi tạo hàm nhúng: Hàm khởi tạo mô hình nhúng
    # Chức năng để đưa văn bản ---> vector
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2"):
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)

class VectoSearch:
    # Khởi tạo hàm tìm kiếm vector: Hàm khởi tạo mô hình tìm kiếm vector
    def __init__(self, data, embedding_function):
        self.db = FAISS.from_documents(data, embedding_function, distance_strategy=DistanceStrategy.COSINE)

    # Hàm tìm kiếm sự tương đồng: Tìm kiếm sự tương đồng của câu hỏi người dùng với dữ liệu trong database
    def similarity_search(self, question: str, k: int = 1):
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context
    
    def similarity_search_with_score(self, question: str, k: int = 1):
        results = self.db.similarity_search_with_score(question, k=k)
        return results

    def save_to_local(self, output_dir):
        """
        Save FAISS database to local directory.
        """
        self.db.save_local(output_dir)
        print(f"FAISS vector database saved to: {output_dir}")

    def load_from_local(vector_store_path, embedding_function):
        """
        Load the locally saved FAISS index.
        """
        try:
            return FAISS.load_local(
                vector_store_path, embeddings=embedding_function, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            raise

# Function to load JSON data and convert to Documents
def load_documents_from_json(json_file_path):
    """
    Load documents from a JSON file.
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    documents = []
    for item in data:
        documents.append(
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {})
            )
        )
    return documents

def main():
    # Path to the JSON file containing data
    json_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "data.json")
    
    # Directory to save the FAISS vector database
    faiss_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

    # Step 1: Load documents from JSON
    documents = load_documents_from_json(json_file_path)
    print(f"Loaded {len(documents)} documents from JSON.")

    # Step 2: Initialize embedding function
    embedding = Embedding().embedding_function

    # Step 3: Create and save FAISS vector database
    vector_search = VectoSearch(documents, embedding)
    vector_search.save_to_local(faiss_output_dir)

    # Step 4: Test similarity search
    query = "CLB về ngôn ngữ và văn hóa"
    result = vector_search.similarity_search(query, k=1)
    print("\nSearch Result:")
    print(result)

if __name__ == "__main__":
    main()
