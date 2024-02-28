from llama_index.core import VectorStoreIndex, SummaryIndex
from typing import Sequence
from llama_index.core.schema import BaseNode
from llama_index.core.readers import Document
from llama_index.embeddings.gemini import GeminiEmbedding

import os
from dotenv import load_dotenv
load_dotenv()




def get_embedding_using_GeminiEmbedding():
    embedding_model = GeminiEmbedding(model_name="models/embedding-001")
    return embedding_model


def get_vector_index_using_VectorStoreIndex(nodes: Sequence[BaseNode],embedding_model) -> VectorStoreIndex:
    vector_index = VectorStoreIndex(nodes=nodes,embed_model=embedding_model)
    return vector_index


def get_summary_index_using_SummaryIndex(nodes: Sequence[BaseNode]) -> SummaryIndex:
    summary_index = SummaryIndex(nodes=nodes)
    return summary_index
# def get_emmeding_using

if __name__ == "__main__":
    pass
