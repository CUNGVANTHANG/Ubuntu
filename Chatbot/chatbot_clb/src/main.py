import os
import streamlit as st # Thư viện tạo giao diện web (thay vì code html, css)
from model import Model
from rag import Embedding, VectoSearch, load_documents_from_json

def setup_page():
    st.set_page_config( 
        page_title="Chat Bot",
        page_icon="💬",  
        layout="centered"
    )

def display_title():
    st.title("💬 Chat Bot")

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Danh sách 11 câu lạc bộ hiện có:\n1. **CLB HEC (HSB Event Club)**: Câu lạc bộ chuyên tổ chức các sự kiện lớn nhỏ trong trường như Orientation Day, workshops kỹ năng mềm, các lễ hội như Tết, Halloween, hội chợ và giao lưu sinh viên quốc tế. Đây là môi trường để phát triển kỹ năng tổ chức, quản lý, và giao tiếp chuyên nghiệp.\n\n2. **CLB VOS (Voice of Students)**: Kênh truyền thông đại diện cho sinh viên HSB, chuyên sản xuất các bài viết, video, và nội dung đa phương tiện để truyền tải thông tin và hình ảnh của trường.\n\n3. **CLB HAC (HSB Acoustic Club)**: Nơi dành cho những bạn yêu thích âm nhạc acoustic nhẹ nhàng, lãng mạn. Câu lạc bộ tổ chức các buổi giao lưu, học hỏi và biểu diễn âm nhạc từ các nhạc cụ như guitar, piano, cajon, và hát.\n\n4. **CLB HVC (HSB Volunteer Club)**: Câu lạc bộ tình nguyện chuyên tổ chức các hoạt động cộng đồng như hỗ trợ học sinh vùng sâu vùng xa, bảo vệ môi trường, và phát động phong trào quyên góp hỗ trợ những hoàn cảnh khó khăn.\n\n5. **CLB HPC (HSB Performance Club)**: Dành cho những bạn trẻ đam mê vũ đạo và nghệ thuật biểu diễn, với các hoạt động nhảy hiện đại, dàn dựng tiết mục và biểu diễn trên sân khấu.\n\n6. **CLB HBT (HSB Basketball Team)**: Đội bóng rổ chính thức của HSB, nơi rèn luyện kỹ năng chơi bóng, tham gia các giải đấu cấp trường và thành phố, đồng thời xây dựng tinh thần đồng đội và kỷ luật.\n\n7. **CLB HM (HSB Model)**: Câu lạc bộ thời trang và người mẫu, nơi các thành viên thể hiện bản thân qua các sự kiện, chụp ảnh họa báo và học hỏi về thời trang.\n\n8. **CLB HBC (HSB Badminton Club)**: Dành cho các bạn yêu thích bộ môn cầu lông, từ người mới chơi đến người có kinh nghiệm thi đấu. Đây là nơi rèn luyện sức khỏe và nâng cao kỹ năng cầu lông.\n\n9. **CLB HFC (HSB Football Club)**: Đội bóng đá chính thức của HSB, tập trung vào rèn luyện kỹ năng và tham gia các giải đấu trong và ngoài trường.\n\n10. **CLB HDC (HSB Debate Club)**: Câu lạc bộ tranh biện, nơi phát triển khả năng lập luận, phản biện, và tư duy logic qua các buổi thảo luận về kinh tế, xã hội, và văn hóa.\n\n11. **CLB GLC (HSB Global Language & Culture Club)**: Câu lạc bộ ngôn ngữ và văn hóa, giúp sinh viên phát triển kỹ năng ngoại ngữ, giao tiếp, và tìm hiểu phong tục các quốc gia trên thế giới.\n\n Bạn cần tư vấn về CLB nào không?"})   

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str, vector_search: VectoSearch, model: Model):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            context = None if not vector_search else vector_search.similarity_search(prompt, k=4)

            restricted_prompt = (
                f"Chỉ trả lời dựa trên nội dung phù hợp sau đây:\n{context}\n"
                f"Nếu nội dung không có thông tin phù hợp, không được trả lời khác hãy trả lời: "
                f'\"Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi này.\"\n'
                f"Câu hỏi của người dùng: {prompt}"
            )

            print(context)

            answer = model.generate(restricted_prompt, context)

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

def main():
    setup_page()
    display_title()
    initialize_chat_history()
    display_chat_messages()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
    json_file_path = os.path.join(base_dir, "data", "data.json")
    vector_store_path = os.path.join(base_dir, "vector_store")

    documents = load_documents_from_json(json_file_path)
    
    # Initialize model
    model = Model()
    embedding = Embedding().embedding_function

    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        print("Vector store không tồn tại. Tạo mới...")
        documents = load_documents_from_json(json_file_path)
        print(f"Loaded {len(documents)} documents from JSON.")
        vector_search = VectoSearch(documents, embedding)
        vector_search.save_to_local(vector_store_path)
    else:
        print("Tải vector store hiện có...")
        vector_search = VectoSearch.load_from_local(vector_store_path, embedding)

    # Handle user input
    if prompt := st.chat_input("Bạn cần tư vấn gì không?"):
        handle_user_input(prompt, vector_search, model)

if __name__ == "__main__":
    main()                   