import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class Model:
    # Hàm khởi tạo: Hàm set up mô hình (model)
    def __init__(self, model_id: str = "gemini-1.5-pro"):
        self.model_id = model_id

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")

        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")

        # Generation and safety configuration
        self.generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

    # Hàm truy vấn API: Hàm gửi yêu cầu tới API để tạo câu trả lời
    def query_gemini_api(self, system_instruction: str):
        model = genai.GenerativeModel(
            model_name=self.model_id,
            safety_settings=self.safety_settings,
            generation_config=self.generation_config,
            system_instruction=system_instruction,
        )
        return model.start_chat(history=[])

    # Hàm tạo câu trả lời: Hàm tạo câu trả lời cho câu hỏi từ người dùng
    def generate(self, question: str, context: str = None):
        # Prepare the prompt
        if not context:
            prompt = f"Question: {question}"
        else:
            prompt = f"Context: {context}\nQuestion: {question}"

        # Define system instruction
        system_instruction = (
            "Bạn là một người tư vấn thông tin cho Câu Lạc Bộ (CLB). Nhiệm vụ của bạn là hỗ trợ người dùng tìm hiểu, cung cấp thông tin, "
            "phân tích và giải đáp các thắc mắc liên quan đến CLB. Hãy cung cấp câu trả lời rõ ràng, chi tiết và dễ hiểu kèm các ví dụ thực tế "
            "(nếu cần) để giúp người dùng nắm được mục đích, hoạt động và các thông tin hữu ích khác của CLB. Không trả lời những thông tin ngoài"
        )

        # Query Gemini API: Gửi request tới API để tạo câu trả lời
        chat_session = self.query_gemini_api(system_instruction=system_instruction)

        # Nhận response từ API
        response = chat_session.send_message(prompt)

        return response.text

def main():
    try:
        model = Model(model_id="gemini-1.5-pro")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("✅ Model đã khởi tạo thành công!")
    
    print("\n💬 Chat Bot: Hãy nhập câu hỏi của bạn. Gõ 'exit' để thoát.")
    
    while True:
        user_input = input("Bạn: ")
        
        if user_input.lower() == "exit":
            print("👋 Tạm biệt! Hẹn gặp lại.")
            break
        
        try:
            answer = model.generate(question=user_input, context=None)
            print(f"\n🤖 Chat Bot:\n{answer}\n")
        except Exception as e:
            print(f"❌ Đã xảy ra lỗi khi gọi Gemini API: {e}")

if __name__ == "__main__":
    main()