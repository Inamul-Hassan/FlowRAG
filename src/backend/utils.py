from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.storage.chat_store import SimpleChatStore

def preprocess(data_dir: str, config: dict):
    documents = SimpleDirectoryReader(input_dir = data_dir).load_data()
    splitter = SentenceSplitter(**config["transform"])
    nodes = splitter.get_nodes_from_documents(documents)
        
    return nodes

def store(nodes, embed_model, vector_config, chat_config, data_description) -> None:
    db = chromadb.PersistentClient(path = vector_config["db_loc"])
    chroma_collection = db.get_or_create_collection(name = vector_config["collection_name"])
    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)

    VectorStoreIndex(
        nodes = nodes, 
        embed_model = embed_model, 
        storage_context = storage_context
    )

    chat_store = SimpleChatStore()
    chat_store.add_message(chat_config["key"], message= ChatMessage(
                role=MessageRole.USER,
                content=f"Hello assistant, we are having a insightful discussion about {data_description}"))
    chat_store.add_message(chat_config["key"], message= ChatMessage(
                role=MessageRole.ASSISTANT, 
                content="Okay, sounds good."))
    chat_store.persist(persist_path=chat_config["loc"]) 

def load(embed_model, db_loc: str, collection_name: str, chat_history_loc: str):
    db = chromadb.PersistentClient(path = db_loc)
    chroma_collection = db.get_or_create_collection(name = collection_name)
    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store = vector_store,
        embed_model = embed_model
    )

    chat_store = SimpleChatStore.from_persist_path(chat_history_loc)
    
    return index, chat_store

def append_chat(store, loc: str, key: str, query: str, response: str):
    store.add_message(key, ChatMessage(role=MessageRole.USER, content = query))
    store.add_message(key, ChatMessage(role=MessageRole.ASSISTANT, content = response))
    store.persist(persist_path = loc)


def get_chat_engine(query_engine, llm, chat_store, config):
    custom_prompt = PromptTemplate("""Given a conversation (between Human and Assistant) and a follow up message from Human, rewrite the message to be a standalone question that captures all relevant context from the conversation and that standalone question can be used to query a vector database to get the relavent data.\n<Chat History>\n{chat_history}\n<Follow Up Message>\n{question}\n<Standalone question>""")

    custom_chat_history = chat_store.get_messages(config["key"]) 

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine = query_engine,
        condense_question_prompt = custom_prompt,
        chat_history = custom_chat_history,
        verbose = True,
        llm = llm
    )

    return chat_engine

import os
import zipfile

from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/download/{file_paths}")
async def zip_and_download(file_paths: str):
    preset = ""
    files_and_folders = f"{preset},{file_paths}".split(",")
    zip_file_path = r"tmp\download.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        for file_or_folder in files_and_folders:
            if os.path.isfile(file_or_folder):
                zipf.write(file_or_folder, os.path.basename(file_or_folder))
            elif os.path.isdir(file_or_folder):
                for root, _, files in os.walk(file_or_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, file_or_folder))

    return FileResponse(zip_file_path, media_type="application/zip", filename="FlowRAG.zip")