from llama_index.core import VectorStoreIndex, SummaryIndex
from typing import Sequence, Optional
from llama_index.core.schema import BaseNode
# from llama_index.core.readers import Document
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import chromadb
from qdrant_client import QdrantClient
from llama_index.core.vector_stores.types import BasePydanticVectorStore

import os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()




def store_using_chromadb(collection_name:str)->BasePydanticVectorStore:
    """
    PresistentClient by default saves the data in the storage/chromadb folder
    """
    db = chromadb.PersistentClient(path="storage/chromadb")
    chroma_collection = db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def store_using_qdrant(collection_name:str,isLocal:bool=True,config:Optional[Dict]=None)->BasePydanticVectorStore:
    if isLocal:
        client = QdrantClient(path="storage/qdrant")
    else:
        client = QdrantClient(**config)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return vector_store

def create_VectorStroreIndex_from_nodes(nodes: Sequence[BaseNode],embedding_model,vector_store:BasePydanticVectorStore)->VectorStoreIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex(nodes=nodes,embed_model=embedding_model,storage_context=storage_context)
    return vector_index

def create_VectorStoreIndex_from_vector_store(vector_store:BasePydanticVectorStore,embedding_model)->VectorStoreIndex:
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store,embed_model=embedding_model)
    return vector_index
    
# FOR REFERENCE SUMMARY INDEX

# create_SummaryIndex_from_nodes(nodes: Sequence[BaseNode],vector_store:BasePydanticVectorStore)->SummaryIndex:
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     summary_index = SummaryIndex(nodes=nodes,storage_context=storage_context)
#     return summary_index

# def get_index_using_SummaryIndex(nodes: Sequence[BaseNode],storage_context:StorageContext=None) -> SummaryIndex:
#     summary_index = SummaryIndex(nodes=nodes,storage_context=storage_context)
#     return summary_index


if __name__ == "__main__":
    pass
