"""
Sub Question Query Engine
https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html
Progress: 
    - All the core mechanics are working (retriever, sub querying)
    - Optimal chunk size is 1000 without overlap
    - chat storage
    - chat with history
    - supprot for thrid party observability tool
TODO:
    - Indexing issue - resolve
    - Try out different retrievers
"""
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.core import SimpleDirectoryReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

import chromadb
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext,load_index_from_storage

from llama_index.core import VectorStoreIndex

from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.settings import Settings

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer

from dotenv import load_dotenv
load_dotenv()

# import llama_index.core
# llama_index.core.set_global_handler("simple")

import phoenix as px
import llama_index.core
llama_index.core.set_global_handler("arize_phoenix")
session = px.launch_app()


# callback
# from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
# llama_debug = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([llama_debug])
# Settings.callback_manager = callback_manager

# model configuration
llm = Gemini(model_name="models/gemini-pro",)
embed_model = GeminiEmbedding(model_name = "models/embedding-001")
Settings.embed_model = embed_model
Settings.llm = llm

class SubQuestionQuerying:
    """
    input_dir: str - input directory containing the documents
    rcts_config = {chunk_size=1000,
                       chunk_overlap=0,
                       seperators = ["\n\n", "\n", " ", ""] }
    
    """
    def __init__(self,input_dir:str,rcts_config:dict):
        """
        input_dir: str - input directory containing the source documents
        rcts_config = {chunk_size=1000,
                       chunk_overlap=0,
                       seperators = ["\n\n", "\n", " ", ""] }
        """
        
        self.documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        self.rcts_config = rcts_config        
        
    # def store_to_chroma(self,collection_name:str)->BasePydanticVectorStore:
    #     """
    #     PresistentClient by default saves the data in the storage/chromadb folder
    #     """
    #     db = chromadb.PersistentClient(path="storage/chromadb")
    #     chroma_collection = db.get_or_create_collection(name=collection_name)
    #     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    #     return vector_store
    
    def query(self,mode:str,query:str):
        """
        mode: str - "create" or "query"
        query: str - query to be executed
        
        """
        match mode:
            case "create":
                # transformation
                recursive_character_text_splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(**self.rcts_config))
                nodes = recursive_character_text_splitter.get_nodes_from_documents(self.documents)
                
                # vector indexing
                vector_index = VectorStoreIndex(nodes=nodes)
                vector_index.set_index_id("paul_graham_essay")
                vector_index.storage_context.persist("storage")
                
                # chat store
                chat_store = SimpleChatStore()
                chat_store.add_message("user1", message= ChatMessage(
                    role=MessageRole.USER,
                    content="Hello assistant, we are having a insightful discussion about Paul Graham today."))
                chat_store.add_message("user1", message= ChatMessage(
                    role=MessageRole.ASSISTANT, 
                    content="Okay, sounds good."))
                    
            case "query":
                # vector index and chat store from local storage
                storage_context = StorageContext.from_defaults(persist_dir="storage")
                vector_index = load_index_from_storage(storage_context, index_id="paul_graham_essay")
                chat_store = SimpleChatStore.from_persist_path(persist_path="storage/chat_store.json")
            case _:
                raise ValueError("Invalid mode")
        # retriever   
        retriever = VectorIndexRetriever(index=vector_index,similarity_top_k=5)
        
        # query engine
        vector_query_engine = RetrieverQueryEngine(retriever=retriever)
        
        # SubQueryEngine
        query_engine_tools = [
            QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="paul_graham_essay",
                description="Paul Graham essay on What I Worked On",
            ),
        ),
        ]
        
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
        )
        
        custom_prompt = PromptTemplate("""Given a conversation (between Human and Assistant) and a follow up message from Human, rewrite the message to be a standalone question that captures all relevant context from the conversation and that standalone question can be used to query a vector database to get the relavent data.\n<Chat History>\n{chat_history}\n<Follow Up Message>\n{question}\n<Standalone question>""")

        custom_chat_history = chat_store.get_messages("user1")    

        # chat engine
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=custom_prompt,
            chat_history=custom_chat_history,
            verbose=True,
            llm=llm
)
        chat_response = chat_engine.chat(query)

        # storing the chat history
        chat_store.add_message("user1", ChatMessage(role=MessageRole.USER, content=query))
        chat_store.add_message("user1", ChatMessage(role=MessageRole.ASSISTANT, content=chat_response.response))
        chat_store.persist(persist_path="storage/chat_store.json")
        
        # for observability tool
        try:
            import time
            while True:
                time.sleep(100)
        except KeyboardInterrupt:
            return (chat_response)
        
        
        

        
        
if __name__ == "__main__":
    sqqe = SubQuestionQuerying(input_dir="storage/data", rcts_config={
        "chunk_size": 1000,
        "chunk_overlap": 0
    })
    
    sqqe.query(mode="query",query="what made the author interested in the AI?")