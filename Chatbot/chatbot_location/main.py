import os
import streamlit as st 
from model_gemini import initialize_model
from embedding import process_faiss_vector_store
def setup_page():
    st.set_page_config( 
        page_title="Chat Bot",
        page_icon="🌍",  
        layout="centered"
    )

def display_title():
    st.title("🌍 Chat Bot")

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt, generate_response):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            json_file_path = os.path.join(os.path.dirname(__file__), "data.json")
            faiss_output_dir = os.path.join(os.path.dirname(__file__), "vector_store")
            context = process_faiss_vector_store(json_file_path, faiss_output_dir, query=prompt)
            print(context)
            answer = generate_response(prompt, context=context)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

def main():
    setup_page()
    display_title()
    initialize_chat_history()
    display_chat_messages()
    generate_response = initialize_model(model_id="gemini-1.5-pro")
    
    # Handle user input
    if prompt := st.chat_input("What place in Vietnam do you want to ask about?"):
        handle_user_input(prompt, generate_response)

if __name__ == "__main__":
    main()
