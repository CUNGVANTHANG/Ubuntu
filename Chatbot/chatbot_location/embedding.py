import os
import json # Thư viện đọc file JSON
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_embedding_function(model_name: str = "sentence-transformers/all-MiniLM-L12-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(data, embedding_function):
    return FAISS.from_documents(data, embedding_function, distance_strategy=DistanceStrategy.COSINE)

def similarity_search(db, question: str, k: int = 1):
    retrieved_docs = db.similarity_search(question, k=k)
    context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
    return context

def save_vector_store_to_local(db, output_dir):
    db.save_local(output_dir)
    print(f"FAISS vector database saved to: {output_dir}")

def load_vector_store_from_local(vector_store_path, embedding_function):
    try:
        return FAISS.load_local(
            vector_store_path, embeddings=embedding_function, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        raise

def load_documents_from_json(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    documents = []
    for item in data:
        page_content = f"Q: {item['question']}\nA: {item['answer']}"
        documents.append(
            Document(
                page_content=page_content,
                metadata={"source": "QA Script"}
            )
        )
    return documents

def process_faiss_vector_store(json_file_path, faiss_output_dir, query):
    embedding_function = initialize_embedding_function()

    # Check if vector store index already exists
    if os.path.exists(faiss_output_dir):
        print("Loading existing FAISS vector store...")
        vector_store = load_vector_store_from_local(faiss_output_dir, embedding_function)
    else:
        print("Creating new FAISS vector store...")
        documents = load_documents_from_json(json_file_path)
        print(f"Loaded {len(documents)} documents from JSON.")

        vector_store = create_vector_store(documents, embedding_function)
        save_vector_store_to_local(vector_store, faiss_output_dir)

    # Perform similarity search
    result = similarity_search(vector_store, query, k=4)
    return result

def main():
    # Đường dẫn đến file JSON chứa dữ liệu
    json_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "data.json")
    
    # Thư mục lưu cơ sở dữ liệu FAISS
    faiss_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

    # Câu truy vấn để kiểm tra tìm kiếm
    query = "Thành phố Hải Phòng nổi tiếng với những địa danh nào?"

    # Thực hiện xử lý toàn bộ quy trình
    process_faiss_vector_store(json_file_path, faiss_output_dir, query)

if __name__ == "__main__":
    main()
