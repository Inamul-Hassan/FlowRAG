from llama_index.core import Settings
from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

import backend.utils as utils

# import phoenix as px
# import llama_index.core
# llama_index.core.set_global_handler("arize_phoenix")
# session = px.launch_app()

class ReciprocalRerankFusionRetriever:

    def __init__(self, data_dir: str, config: dict, llm: LLM, embed_model: BaseEmbedding):
                
        self.nodes = utils.preprocess(data_dir = data_dir, config = config)
        self.config = config
        self.llm = llm
        self.embed_model = embed_model

    def store(self) -> None:
        utils.store(
            nodes = self.nodes,
            embed_model = self.embed_model,
            vector_config = self.config["storage"], 
            chat_config = self.config["chat_history"]
        )

    def query(self, query_str: str, debug: bool = False) -> str:
 
        index, chat_store = utils.load(
            embed_model = self.embed_model,
            db_loc = self.config["storage"]["db_loc"], 
            collection_name = self.config["storage"]["collection_name"], 
            chat_history_loc = self.config["chat_history"]["loc"]
        )

        base_retriever = index.as_retriever(similarity_top_k = self.config["retriever"]["similarity_top_k"])
        bm25_retriever = BM25Retriever.from_defaults(nodes = self.nodes, similarity_top_k = self.config["retriever"]["similarity_top_k"])

        retriever = QueryFusionRetriever(
            **self.config["retriever"], 
            retrievers = [base_retriever, bm25_retriever],
            mode = "reciprocal_rerank",
        )    

    
        query_engine = RetrieverQueryEngine.from_args(retriever)
    
        chat_engine = utils.get_chat_engine(
            query_engine = query_engine,
            llm = self.llm,
            chat_store = chat_store,
            config = self.config["chat_history"]
        )
        
        response = chat_engine.chat(query_str).response

        utils.append_chat(
            store = chat_store,
            loc = self.config["chat_history"]["loc"],
            key = self.config["chat_history"]["key"],
            query = query_str,
            response = response
        )

        # if debug:
        #     import time
        #     while True:
        #         time.sleep(100)

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
            },
            "chat_history": {
                "loc": "storage/chat_store.json",
                "key": "user01"
            }
        },
        llm = llm,
        embed_model = embed_model
    )
    rag.store()

    print(rag.query("Does he applied to two art schools?", debug = False))
    print(rag.query("What are those?", debug = False))
    print(rag.query("Which one is the first?", debug=False))
    print(rag.query("Which one comes second?", debug=True))