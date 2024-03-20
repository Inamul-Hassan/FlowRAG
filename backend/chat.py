# import sys
# # sys.path.append(r'C:\Users\vishal\Documents\AI\RAG pipeline\FlowRAG')
# sys.path.append("E:\Python\Directory\FlowRAG")

import streamlit as st
from llama_index.core.llms import ChatMessage, MessageRole
from pathlib import Path

from subquestionqueryengine import SubQuestionQuerying
from reciprocalrerankfusionretriever import ReciprocalRerankFusionRetriever
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
import json
from dotenv import load_dotenv
load_dotenv()

print("Imported all modules...")


llm = Gemini(model_name="models/gemini-pro")
embed_model = GeminiEmbedding(model_name = "models/embedding-001")
Settings.embed_model = embed_model
Settings.llm = llm

print("Loaded Llms...")
# llm = "llm"
# embed_model = "embed_model"

def load_json(file_path):
  with open(file_path) as f:
    data = json.load(f)
  return data

user_config = load_json('user_config.json')
pipeline = user_config["pipline"]


print("Loading RAG...")

if "rag" not in st.session_state: 
    match pipeline:
        case "SubQuestionQuerying":
            rag = SubQuestionQuerying(data_dir="storage/data",
                                      config=user_config["config"],
                                      llm=llm,
                                      embed_model=embed_model)
        case "ReciprocalRerankFusionRetriever":
            rag = ReciprocalRerankFusionRetriever(data_dir="storage/data",
                                                  config=user_config["config"],
                                                  llm=llm,
                                                  embed_model=embed_model)
        case _:
            raise ValueError("Invalid pipeline selected")
    # rag = SubQuestionQuerying(
    #         data_dir = "storage/data",
    #         config = {
    #             "transform": {
    #                 "chunk_size": 512
    #             },
    #             "retriever": {
    #                 "similarity_top_k": 5,
    #                 "num_queries": 4
    #             },
    #             "storage": {
    #                 "db_loc": "storage/chromadb",
    #                 "collection_name": "defaultDB"
    #             },
    #             "chat_history": {
    #                 "loc": "storage/chat_store.json",
    #                 "key": "user01"
    #             }
    #         },
    #         llm = llm,
    #         embed_model = embed_model
    #     )
    rag.store()
    st.session_state.rag = rag
    
print("Started!")


def onClick():
  st.session_state.submitted = True

# Steeamlit page configuration
st.set_page_config(page_title="FlowRAG", page_icon="🧠", layout="wide")
st.title("FlowRAG - chat with your documents")
# Initialize chat history only if it doesn't exist, so only once in a session it will be initialized
# This is a way to counter the fact that Streamlit reruns the whole script on every user interaction


# sidebar
with st.sidebar:
    st.write("# Configuration")
    llm_mapping = {
      "Gemini":["gemini-pro-1.0"],
      "OpenAI":["gpt3.5-turbo","gpt-4"]
    }
    embedding_models = ["embedding-001","embedding-002","embedding-003"]

    st.write("## LLM")

    llm_provider = st.selectbox(label="Select a llm provider",options=llm_mapping.keys(),index=0)
    options = llm_mapping[llm_provider]
    llm_model = st.selectbox(label="Select a llm model",options=options,index=0)
    llm_api_key = st.text_input(label="API Key",type="password")
    
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
      
    if llm_api_key is None or llm_api_key == "" or st.session_state.file_uploaded == False:
      st.button(label="Get Starteds",disabled=True)
    else:
      st.button(label="Get Started",on_click=onClick)
      
    st.write(st.session_state)

if "submitted" not in st.session_state:
    st.info("Please provide your API key to get started.")
    st.write(st.session_state)
else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [ChatMessage(role=MessageRole.ASSISTANT, content="Hi. I am your personal assistant. I will try to answer your questions from the provided documents.")]    

    # chat info
    # st.info(f"Chatting with the website: {website_url}")

    # user input
    user_query = st.chat_input(placeholder="Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(ChatMessage(role=MessageRole.USER,content=user_query))
        
        # response = rag.query(user_query, debug = False)
        response = st.session_state.rag.query(user_query, debug = False)
        
        st.session_state.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=response))
        # st.write(st.session_state.chat_history)

    # chat
    for message in st.session_state.chat_history:
        if (message.role == MessageRole.ASSISTANT):
            with st.chat_message(name="AI"):
                st.write(message.content)
        elif (message.role == MessageRole.USER):
            with st.chat_message(name="Human"):
                st.write(message.content)

    # debug
    with st.expander(label="Debug"):
        st.write(st.session_state.chat_history)

        st.write(user_query)

    # with st.chat_message(name="AI"):
    #     st.write(
    #         "Hello! I'm DeepChat. Paste a website URL in the sidebar to get started.")

    # with st.chat_message(name="Human"):
    #     st.write("Hi! I'm excited to chat with you.")
