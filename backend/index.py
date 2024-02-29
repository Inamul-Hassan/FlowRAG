from llama_index.core import VectorStoreIndex, SummaryIndex
from typing import Sequence, Optional
from llama_index.core.schema import BaseNode
# from llama_index.core.readers import Document
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import chromadb
from qdrant_client import QdrantClient

import os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()


def save_to_chromadb(collection_name:str)->StorageContext:
    db = chromadb.PersistentClient(path="storage/chromadb")
    chroma_collection = db.get_or_create_collection(name=collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context
    
def save_to_qdrant(collection_name:str,isLocal:bool=True,config:Optional[Dict]=None)->StorageContext:
    if isLocal:
        client = QdrantClient(location=":memory:",path="storage/qdrant")
    else:
        client = QdrantClient(**config)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def get_index_using_VectorStoreIndex(nodes: Sequence[BaseNode],embedding_model,storage_context:StorageContext=None) -> VectorStoreIndex:
    vector_index = VectorStoreIndex(nodes=nodes,embed_model=embedding_model,storage_context=storage_context)
    return vector_index

def get_index_using_SummaryIndex(nodes: Sequence[BaseNode],storage_context:StorageContext=None) -> SummaryIndex:
    summary_index = SummaryIndex(nodes=nodes,storage_context=storage_context)
    return summary_index

# def store_locally(index):
#     index.


if __name__ == "__main__":
    pass
