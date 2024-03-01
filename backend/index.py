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

"""
TODO
Have to refactor the code
-> Current implementation
   we have two seperate functions for saving and loading from databases.
   But while loading we dont support other indexing methods
   either we have to support that or think of a better solution
-> Better solution
   have seperate functions for each indexing method to create index from vector_store and nodes
   if created from nodes, it will take in nodes and storage_context(if applicable)
   if created from vector_store, it will take in vector_store
"""



def save_to_chromadb(collection_name:str)->StorageContext:
    """
    PresistentClient by default saves the data in the storage/chromadb folder
    """
    db = chromadb.PersistentClient(path="storage/chromadb")
    chroma_collection = db.get_or_create_collection(name=collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def load_from_chromadb(collection_name:str,embedding_model)->BasePydanticVectorStore:
    db = chromadb.PersistentClient(path="storage/chromadb")
    chroma_collection = db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store,embed_model=embedding_model)
    return vector_index
    
def save_to_qdrant(collection_name:str,isLocal:bool=True,config:Optional[Dict]=None)->StorageContext:
    if isLocal:
        client = QdrantClient(path="storage/qdrant")
    else:
        client = QdrantClient(**config)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def load_from_qdrant(collection_name:str,embedding_model,isLocal:bool=True,config:Optional[Dict]=None)->StorageContext:
    if isLocal:
        client = QdrantClient(path="storage/qdrant")
    else:
        client = QdrantClient(**config)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store,embed_model=embedding_model)
    return vector_index


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
