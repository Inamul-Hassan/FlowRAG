"""
Multi-Step Query Engine
https://docs.llamaindex.ai/en/stable/examples/query_transformations/SimpleIndexDemo-multistep.html

TODO:
- Thid method is not working as intended.
- Potential issues
    - The new query generation is not good with gemini
    - The propmt to get the new query can improved with R&D
    - Retruiever is not working properly
- Solution
    - Will rewrite the logic to get the new query
    - optimize the propmt
    - try using different retriever
    
"""

from flowrag.models.llm import chat_using_Gemini
from flowrag.models.embedding import embed_using_GeminiEmbedding
from flowrag.load import load_using_SimpleDirectoryReader
from flowrag.index import create_VectorStroreIndex_from_nodes

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core.query_engine import MultiStepQueryEngine

from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
    DecomposeQueryTransform,
    HyDEQueryTransform
    
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.settings import Settings

# model configuration
llm = chat_using_Gemini()
embedding_model = embed_using_GeminiEmbedding()
Settings.embed_model = embedding_model
Settings.llm = llm

# load documents
documents = load_using_SimpleDirectoryReader(input_dir="storage/data")
index = VectorStoreIndex.from_documents(documents)

# transform documents
step_decompose_transform = HyDEQueryTransform(llm)


index_summary = "Used to answer questions about the author"

query_engine = index.as_query_engine(llm=llm)

query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary
)
response = query_engine.query(
    "Who was in the first batch of the accelerator program the author"
    " started?",
)

print(response)
 