import streamlit as st
import json
from pathlib import Path

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(page_title="FlowRAG", page_icon="💬")
st.title("FlowRAG")
st.write("An Advance RAG Pipeline For Your Data")

# hide side bar
st.markdown("""
      <style>
          section[data-testid="stSidebar"][aria-expanded="true"]{
              display: none;
          }
      </style>
      """, unsafe_allow_html=True)

def load_json(file_path):
      with open(file_path) as f:
        data = json.load(f)
      return data

user_config = load_json('src/pages/user_config.json')
pipeline = user_config["pipline"]

def onClick():
      st.session_state.submitted = True


if "submitted" not in st.session_state:
    # get api key and data
    st.warning("Please provide your API key and data to get started.")
    llm_mapping = {
      "Gemini":["models/gemini-pro"],
    }
    embedding_models = ["models/embedding-001"]

    st.session_state.llm_provider = st.selectbox(label="Select a llm provider",options=llm_mapping.keys(),index=0)
    options = llm_mapping[st.session_state.llm_provider]
    st.session_state.llm_model = st.selectbox(label="Select a llm model",options=options,index=0)
    st.session_state.llm_api_key = st.text_input(label="API Key",type="password")
    
    if "file_uploaded" not in st.session_state:
      st.session_state.file_uploaded = False
    
    files = st.file_uploader(label="Upload a file",type=["pdf","txt"],accept_multiple_files=True)
    save_folder = 'storage/data'
    for file in files:
      if file is not None:
        save_path = Path(save_folder, file.name)
        with open(save_path, mode='wb') as w:
            w.write(file.getvalue())
        st.session_state.file_uploaded = True
    st.session_state.data_description = st.text_input(label="Write a small description about your data",max_chars=200)
      
    if st.session_state.llm_api_key is None or st.session_state.llm_api_key == "" or st.session_state.file_uploaded == False or st.session_state.data_description is None or st.session_state.data_description == "":
      st.button(label="Get Started",disabled=True)
    else:
      st.button(label="Get Started",on_click=onClick)
      
    st.write(st.session_state)

else:
    with st.spinner("Hang On, We are setting up the environment..."):
      # initialize the RAG pipeline
      from backend.subquestionqueryengine import SubQuestionQuerying
      from backend.reciprocalrerankfusionretriever import ReciprocalRerankFusionRetriever
      
      from llama_index.core.llms import ChatMessage, MessageRole
      from llama_index.llms.gemini import Gemini
      from llama_index.embeddings.gemini import GeminiEmbedding
      
      # its hard-coded now change in future
      llm = Gemini(model_name="models/gemini-pro",api_key=st.session_state.llm_api_key)
      embed_model = GeminiEmbedding(model_name = "models/embedding-001",api_key=st.session_state.llm_api_key)
      
      if "rag" not in st.session_state: 
        match pipeline:
            case "SubQuestionQuerying":
                rag = SubQuestionQuerying(data_dir="storage/data",
                                          data_description=st.session_state.data_description,
                                          config=user_config["config"],
                                          llm=llm,
                                          embed_model=embed_model)
            case "ReciprocalRerankFusionRetriever":
                rag = ReciprocalRerankFusionRetriever(data_dir="storage/data",
                                                      data_description=st.session_state.data_description,
                                                      config=user_config["config"],
                                                      llm=llm,
                                                      embed_model=embed_model)
            case _:
                raise ValueError("Invalid pipeline selected")
        rag.store()
        st.session_state.rag = rag
      
      # session state
      if "chat_history" not in st.session_state:
          st.session_state.chat_history = [ChatMessage(role=MessageRole.ASSISTANT, content="Hi. I am your personal assistant. I will try to answer your questions from the provided documents.")]    

    # chat info
    st.info(f"This chatbot is based on: {pipeline}")
    
    # chat
    for message in st.session_state.chat_history:
        if (message.role == MessageRole.ASSISTANT):
            with st.chat_message(name="AI"):
                st.write(message.content)
        elif (message.role == MessageRole.USER):
            with st.chat_message(name="Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input(placeholder="Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(ChatMessage(role=MessageRole.USER,content=user_query))
        
        with st.chat_message(name="Human"):
            st.write(user_query)
        with st.spinner("Thinking..."):
            response = st.session_state.rag.query(user_query)
        with st.chat_message(name="AI"):
            st.write(response)
        
        st.session_state.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=response))
