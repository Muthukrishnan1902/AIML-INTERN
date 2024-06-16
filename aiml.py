import os
import tempfile
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import CTranslate2
import streamlit as st

# Set the environment variable to avoid OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize the language model
model_id = 'NHL2-13b-chat-Llama2-ct2'
tokenizer_id = 'Nous-Hermes-Llama2-13b'
llm = CTranslate2(model_path=model_id, tokenizer_name=tokenizer_id, device="auto", compute_type="int8")

# Configure Streamlit page
st.set_page_config(page_title="Chat with Llama-3", layout="wide")
st.header("Chat with Llama-3 using your CSV")

# Allow the user to upload a CSV file
file = st.file_uploader("Upload your CSV file", type="csv")

if file is not None:
    # Create a temporary file to store the uploaded CSV data
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as f:
        # Convert bytes to a string before writing to the file
        data_str = file.getvalue().decode('utf-8')
        f.write(data_str)
        f.flush()

        # Create a CSV agent using the language model and the temporary file
        agent = create_csv_agent(llm, f.name, verbose=True, agent_executor_kwargs={"handle_parsing_error": True})

        # Display a chat interface
        st.write("### Chat Interface")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        while True:
            # Display the chat history
            for chat in st.session_state.chat_history:
                st.write(f"You: {chat['user_input']}")
                st.write(f"Llama-3: {chat['response']}")

            # Ask the user to input a question
            user_input = st.text_input("Enter your question here:")

            if st.button("Send"):
                if user_input:
                    # Run the agent on the user's question and get the response
                    response = agent.run(user_input)
                    # Store the interaction in the chat history
                    st.session_state.chat_history.append({'user_input': user_input, 'response': response})
                    # Clear the input field
                    st.experimental_rerun()
