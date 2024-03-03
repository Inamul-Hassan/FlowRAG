from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(r'C:\Users\vishal\Documents\AI\RAG pipeline\FlowRAG\backend')


from flowrag.load import load_using_SimpleDirectoryReader
from flowrag.transform import transform_using_SentenceSplitter
from llama_index.core import VectorStoreIndex

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.base_query_engine import BaseQueryEngine

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

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
        }
        '''
        self.data_dir = data_dir
        self.config = config
        self.llm = llm
        self.embed_model = embed_model

    def query(self, query_str: str) -> BaseQueryEngine:
        documents = load_using_SimpleDirectoryReader(input_dir = self.data_dir)
        nodes = transform_using_SentenceSplitter(documents = documents, config = self.config["transform"])

        index = VectorStoreIndex(nodes = nodes, embed_model = self.embed_model)
        retriever = index.as_retriever(similarity_top_k = self.config["retriever"]["similarity_top_k"])
        bm25_retriever = BM25Retriever.from_defaults(index = index, similarity_top_k = self.config["retriever"]["similarity_top_k"])

        retriever = QueryFusionRetriever(
            **self.config["retriever"], 
            retrievers = [retriever, bm25_retriever],
            mode = "reciprocal_rerank"
        )    
        query_engine = RetrieverQueryEngine.from_args(retriever)

        return query_engine.query(query_str)