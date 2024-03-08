from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.base import BaseIndex

from llama_index.retrievers.bm25 import BM25Retriever

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.core import VectorStoreIndex

import phoenix as px
import llama_index.core
llama_index.core.set_global_handler("arize_phoenix")
session = px.launch_app()

class ReciprocalRerankFusionRetriever:

    def __init__(self, data_dir: str, config: dict, llm: LLM, embed_model: BaseEmbedding):
        '''
        config = {
            "transform": {
                "chunk_size": 256
            },
            "retriever": {
                "similarity_top_k": 2,
                "num_queries": 4
            }
            "storage": {
                "db_loc": "storage/chromadb",
                "collection_name": "defaultDB"
            }
        }
        '''
        def preprocess(data_dir: str, config: dict):
            documents = SimpleDirectoryReader(input_dir = data_dir).load_data()
            splitter = SentenceSplitter(**config["transform"])
            nodes = splitter.get_nodes_from_documents(documents)
            return nodes
        
        self.nodes = preprocess(data_dir = data_dir, config = config)
        self.config = config
        self.llm = llm
        self.embed_model = embed_model
    
    def store(self) -> None:

        # Initialize ChromaDB
        db = chromadb.PersistentClient(path = self.config["storage"]["db_loc"])
        chroma_collection = db.get_or_create_collection(name = self.config["storage"]["collection_name"])
        vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store = vector_store)

        # Store the nodes to DB and create index out of it
        VectorStoreIndex(
            nodes = self.nodes, 
            embed_model = self.embed_model, 
            storage_context = storage_context
        )

    def query(self, query_str: str, debug: bool) -> str:

        def load(db_loc: str, collection_name: str) -> BaseIndex:
            db = chromadb.PersistentClient(path = db_loc)
            chroma_collection = db.get_or_create_collection(name = collection_name)
            vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store = vector_store,
                embed_model = self.embed_model
            )
            return index

        # Load index from storage
        index = load(db_loc = self.config["storage"]["db_loc"], collection_name = self.config["storage"]["collection_name"])

        # Retrievers
        base_retriever = index.as_retriever(similarity_top_k = self.config["retriever"]["similarity_top_k"])
        bm25_retriever = BM25Retriever.from_defaults(nodes = self.nodes, similarity_top_k = self.config["retriever"]["similarity_top_k"])

        retriever = QueryFusionRetriever(
            **self.config["retriever"], 
            retrievers = [base_retriever, bm25_retriever],
            mode = "reciprocal_rerank",
        )    

        # Query engine
        query_engine = RetrieverQueryEngine.from_args(retriever)
        
        if debug:
            import time
            while True:
                time.sleep(100)

        response =  query_engine.query(query_str)

        return response

if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    llm = Gemini(model_name="models/gemini-pro")
    embed_model = GeminiEmbedding(model_name = "models/embedding-001")
    Settings.embed_model = embed_model
    Settings.llm = llm

    rag = ReciprocalRerankFusionRetriever(
        data_dir = "storage/data",
        config = {
            "transform": {
                "chunk_size": 512
            },
            "retriever": {
                "similarity_top_k": 5,
                "num_queries": 4
            },
            "storage": {
                "db_loc": "storage/chromadb",
                "collection_name": "defaultDB"
            }
        },
        llm = llm,
        embed_model = embed_model
    )
    # rag.store()

    print(rag.query("When did he visit Rich Draves?", debug = True))