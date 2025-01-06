import os
import google.generativeai as genai
from dotenv import load_dotenv

def initialize_model(model_id: str = "gemini-1.5-pro"):
    """Hàm khởi tạo và trả về đối tượng model đã được cấu hình."""
    load_dotenv()  # Load environment variables from .env file
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")

    # Configure Gemini API
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")

    # Generation and safety configuration
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

    def query_gemini_api(system_instruction: str):
        """Hàm gửi yêu cầu tới Gemini API để bắt đầu phiên trò chuyện."""
        model = genai.GenerativeModel(
            model_name=model_id,
            safety_settings=safety_settings,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )
        return model.start_chat(history=[])

    def generate(question: str, context: str = None):
        """Hàm tạo câu trả lời dựa trên câu hỏi của người dùng."""
        # Prepare the prompt
        if not context:
            prompt = f"Question: {question}"
        else:
            prompt = f"Only reply to content within context: {context}\nQuestion: {question}"

        # Define system instruction
        system_instruction = (
            "Bạn là một chatbot tư vấn thông tin về các địa điểm tại Việt Nam. Nhiệm vụ của bạn là cung cấp thông tin, "
            "giải đáp thắc mắc về các địa danh, văn hóa, lịch sử và các thông tin hữu ích khác tại các tỉnh thành ở Việt Nam. "
            "Hãy đưa ra câu trả lời chi tiết, chính xác và dễ hiểu để giúp người dùng khám phá Việt Nam một cách tốt nhất. "
            "Không trả lời những thông tin không liên quan đến địa điểm tại Việt Nam."
        )

        # Query Gemini API: Gửi request tới API để tạo câu trả lời
        chat_session = query_gemini_api(system_instruction=system_instruction)

        # Nhận response từ API
        response = chat_session.send_message(prompt)

        return response.text

    return generate

def chat_with_model():
    """Hàm để tương tác với người dùng qua giao diện console."""
    try:
        generate_response = initialize_model(model_id="gemini-1.5-pro")
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\n✅ Model đã khởi tạo thành công!")
    print("\n💬 Chat Bot: Hãy nhập câu hỏi của bạn. Gõ 'exit' để thoát.")

    while True:
        user_input = input("Bạn: ")

        if user_input.lower() == "exit":
            print("👋 Tạm biệt! Hẹn gặp lại.")
            break

        try:
            answer = generate_response(question=user_input, context=None)
            print(f"\n🤖 Chat Bot:\n{answer}\n")
        except Exception as e:
            print(f"❌ Đã xảy ra lỗi khi gọi Gemini API: {e}")

if __name__ == "__main__":
    chat_with_model()
