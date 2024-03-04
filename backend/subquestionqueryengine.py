"""
Sub Question Query Engine
https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html
Progress: 
    - All the core mechanics are working (retriever, sub querying)
    - Optimal chunk size is 1000 without overlap
TODO:
    - Add chat storage
    - Add chat with history
    - Add supprot for thrid party observability tool


"""

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.core import SimpleDirectoryReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

import chromadb
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.core import VectorStoreIndex

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.settings import Settings

# Logging input and output
import llama_index.core
llama_index.core.set_global_handler("simple")

# Callback
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

# model configuration
llm = Gemini(model_name="models/gemini-pro")
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
        
        self.documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        self.rcts_config = rcts_config
        
    def store_using_chromadb(self,collection_name:str)->BasePydanticVectorStore:
        """
        PresistentClient by default saves the data in the storage/chromadb folder
        """
        db = chromadb.PersistentClient(path="storage/chromadb")
        chroma_collection = db.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
        
    
    def query(self,query:str):
        """
        rcts_config = {chunk_size=1000,
                       chunk_overlap=0,
                       seperators = ["\n\n", "\n", " ", ""] }
        """
        recursive_character_text_splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(**self.rcts_config))
        nodes = recursive_character_text_splitter.get_nodes_from_documents(self.documents)
        
        vector_store = self.store_using_chromadb(collection_name="paul_graham_essay")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        vector_index = VectorStoreIndex(nodes=nodes,storage_context=storage_context)
        
        vector_query_engine = vector_index.as_query_engine()
        
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
            verbose=True
        )
        

        response = query_engine.query(query)

        return response

        
        
if __name__ == "__main__":
    sqqe = SubQuestionQuerying(input_dir="storage/data", rcts_config={
        "chunk_size": 1000,
        "chunk_overlap": 0
    })
    
    sqqe.query("What is the most distinctive thing about YC?")
