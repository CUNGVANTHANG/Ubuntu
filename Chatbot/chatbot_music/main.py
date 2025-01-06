import os # Thư viện hệ thống có tác dùng tương tác với hệ thống máy tính (Dùng để đọc file csv, lưu file cache, lưu file vector_store)
import json # Thư viện xử lý JSON
import re # Thư viện xử lý chuỗi (Dùng để trích xuất link từ câu trả lời)
import streamlit as st # Thư viện tạo giao diện web (Thay vì các bạn code html, css...)
import google.generativeai as genai # Thư viện tương tác với API Gemini
import pandas as pd # Thư viện xử lý dữ liệu dạng bảng (Dùng để đọc file csv)
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CACHE_FILE = "user_cache.json"

# ----------------------------- Rag -----------------------------
# Hàm khởi tạo embedding
def initialize_embedding(model_name: str = "sentence-transformers/all-MiniLM-L12-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

# Hàm tạo vector search (Để tìm kiếm tương tự, tìm kiếm vector)
def create_vector_search(data, embedding_function, distance_strategy=DistanceStrategy.COSINE):
    return FAISS.from_documents(data, embedding_function, distance_strategy=distance_strategy)

# Hàm tìm kiếm tương tự
def similarity_search(vector_store, question: str, k: int = 1):
    retrieved_docs = vector_store.similarity_search(question, k=k)
    context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
    return context

# Hàm tải dữ liệu từ file csv 
def load_csv_as_documents(file_path):
    """Load CSV và chuyển đổi thành danh sách Document."""
    data = pd.read_csv(file_path)
    documents = [
        Document(
            page_content=(
                f"Song Title: {row['Song_Title']}, "
                f"Artist: {row['Artist']}, "
                f"Genre: {row['Genre']}, "
                f"Mood: {row['Mood']}, "
                f"Situation: {row['Situation']}, "
                f"Energy Level: {row['Energy_Level']}, "
                f"Release Year: {row['Release_Year']}, "
                f"Link: {row['Link_URL']}"
            ),
            metadata={"Song_ID": row["Song_ID"]}
        )
        for _, row in data.iterrows()
    ]
    return documents

# Hàm lưu vector store vào local
def save_vector_store_to_local(vector_store, output_dir):
    try:
        vector_store.save_local(output_dir)
        print(f"✅ FAISS vector database saved to: {output_dir}")
    except Exception as e:
        print(f"❌ Error saving vector store: {e}")
        raise

# Hàm tải vector store từ local
def load_vector_store_from_local(vector_store_path, embedding_function):
    try:
        vector_store = FAISS.load_local(
            vector_store_path, embeddings=embedding_function, allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS vector database loaded from: {vector_store_path}")
        return vector_store
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        raise

# ----------------------------- Cache -----------------------------
# Hàm lưu thông tin người dùng vào file cache
def save_user_to_cache(user_data):
    """Lưu thông tin người dùng vào file cache."""
    with open(CACHE_FILE, "w") as f:
        json.dump(user_data, f)

# Hàm tải thông tin người dùng từ file cache
def load_user_from_cache():
    """Tải thông tin người dùng từ file cache."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return None

# Hàm thêm câu hỏi của người dùng vào cache
def add_query_to_cache(user_query):
    """Thêm một câu hỏi của người dùng vào danh sách cache."""
    cache_data = load_user_from_cache() or {}
    if "user_queries" not in cache_data:
        cache_data["user_queries"] = []
    cache_data["user_queries"].append(user_query)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f)

# ----------------------------- Gemini API -----------------------------
# Cấu hình setup model
def configure_gemini_api(model_id: str = "gemini-1.5-pro"):
    """Cấu hình API Gemini và kiểm tra khóa API."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
    
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    system_instruction = (
        "Bạn là MusicMate, một trợ lý âm nhạc thông minh. Nhiệm vụ của bạn là trả lời bằng tiếng việt hỗ trợ người dùng tìm kiếm bài hát, gợi ý playlist, "
        "và cung cấp thông tin liên quan đến âm nhạc. Hãy đưa ra câu trả lời rõ ràng, chi tiết và phù hợp với sở thích, tình huống nghe nhạc, "
        "và tâm trạng của người dùng. Nếu cần, hãy gợi ý các bài hát hoặc playlist cụ thể để người dùng có trải nghiệm âm nhạc tốt nhất."
    )

    model = genai.GenerativeModel(
        model_name=model_id,
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction=system_instruction
    )
    
    return model

# Nhận request từ người dùng và trả về câu trả lời
def generate_response(question: str, model, context: str = None):
    if not context:
        prompt = f"Question: {question}"
    else:
        prompt = f"Context: {context}\nQuestion: {question}"

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    return response.text

# ----------------------------- Giao diện Web -----------------------------
# Hàm cấu hình trang web
def setup_page():
    st.set_page_config(
        page_title="MusicMate",
        page_icon="🎶",
        layout="centered"
    )

# Hàm hiển thị tiêu đề
def display_title():
    st.title("🎶 MusicMate")
    st.subheader("Người bạn đồng hành âm nhạc của bạn!")

# Hàm thu thập thông tin người dùng
def gather_user_info():
    st.write(
        "Chào bạn! Mình là **MusicMate**, người bạn đồng hành âm nhạc của bạn. "
        "Mình sẽ giúp bạn tìm những bài hát và playlist tuyệt vời để thư giãn, làm việc, hay nâng cao tâm trạng!"
    )
    st.write("Trước tiên, mình muốn biết một chút thông tin về bạn để đề xuất nhạc phù hợp nhé!")

    user_name = st.text_input("Bạn có thể cho mình biết tên của bạn được không?", placeholder="Nhập tên của bạn")
    favorite_genre = st.text_input(
        "Bạn thích thể loại nhạc nào? (Ví dụ: Pop, Rock, EDM, Hiphop, Ballad, v.v.)",
        placeholder="Nhập thể loại nhạc bạn thích"
    )
    listening_context = st.text_input(
        "Bạn thường nghe nhạc trong những tình huống nào? (Làm việc, thư giãn, tập thể dục, v.v.)",
        placeholder="Nhập tình huống bạn thường nghe nhạc"
    )
    user_mood = st.text_input(
        "Tâm trạng hiện tại của bạn là gì? (Vui vẻ, mệt mỏi, căng thẳng, v.v.)",
        placeholder="Nhập tâm trạng hiện tại của bạn"
    )
    if st.button("Gửi thông tin"):
        if user_name:
            user_data = {
                "name": user_name,
                "favorite_genre": favorite_genre,
                "listening_context": listening_context,
                "user_mood": user_mood
            }
            save_user_to_cache(user_data) 
            st.session_state["chat_started"] = True
            st.success("Thông tin của bạn đã được lưu thành công!")
        else:
            st.error("Vui lòng nhập tên của bạn trước khi gửi thông tin.")

# Hàm khởi tạo lịch sử chat
def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.audio = []

# Hàm hiển thị tin nhắn chat
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Hàm trích xuất và phát các link nhạc từ câu trả lời
def extract_and_play_links(response):
    links = re.findall(r'https?://\S+', response)

    if links:
        for link in links:
            if "soundcloud.com" in link:
                # Nhúng trình phát bằng iframe
                embed_link = link.replace("https://soundcloud.com", "https://w.soundcloud.com/player/?url=https://soundcloud.com")
                iframe_html = f"""
                <iframe width="100%" height="166" scrolling="no" frameborder="no" 
                        allow="autoplay" 
                        src="{embed_link}">
                </iframe>
                """
                st.markdown(iframe_html, unsafe_allow_html=True)

# Hàm xử lý input từ người dùng
def handle_user_input(cached_user, user_input, model, vector_store):
    st.session_state.messages.append({"role": "user", "content": user_input})
    add_query_to_cache(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            personalized_context = ""
            if cached_user:
                personalized_context = (
                    f"Tên: {cached_user.get('name', 'Người dùng')}\n"
                    f"Thể loại nhạc yêu thích: {cached_user.get('favorite_genre', 'Chưa rõ')}\n"
                    f"Tâm trạng: {cached_user.get('user_mood', 'Chưa rõ')}\n"
                    f"Tình huống nghe nhạc: {cached_user.get('listening_context', 'Chưa rõ')}\n"
                )

            context_from_vector_store = similarity_search(vector_store, user_input, k=10)

            print(context_from_vector_store)

            restricted_prompt = (
                f"Thông tin cá nhân hóa:\n{personalized_context}\n\n"
                f"Chỉ trả lời dựa trên nội dung phù hợp sau đây:\n{context_from_vector_store}\n"
                f"Nghe tại đây: [Link] nếu có.\n"
                f"Nếu nội dung không có thông tin phù hợp, hãy trả lời dựa trên hiểu biết thông thường:\n"
                f"Câu hỏi của người dùng: {user_input}"
            )

            assistant_response = generate_response(user_input, model, restricted_prompt)
            extract_and_play_links(assistant_response)
            st.markdown(assistant_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
# ----------------------------- Main Function -----------------------------
def main():
    setup_page()
    display_title()
    cached_user = load_user_from_cache()

    if not cached_user or not cached_user.get("name"):
        gather_user_info()
    else:
        initialize_chat_history()
        display_chat_messages()
        model = configure_gemini_api("gemini-1.5-pro")
        embedding_function = initialize_embedding()
        vector_store_path = "vector_store"

        if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            csv_file = "songs.csv"
            if not os.path.exists(csv_file):
                st.error("File songs.csv không tồn tại. Vui lòng kiểm tra lại.")
                return
            documents = load_csv_as_documents(csv_file)
            vector_store = create_vector_search(documents, embedding_function)
            save_vector_store_to_local(vector_store, vector_store_path)
        else:
            vector_store = load_vector_store_from_local(vector_store_path, embedding_function)
        if user_input := st.chat_input("Nhập tin nhắn của bạn?"):
            handle_user_input(cached_user, user_input, model, vector_store)
if __name__ == "__main__":
    main()
