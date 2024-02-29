from typing import Dict
from llama_index.core.readers import Document

from llama_index.core.node_parser import SentenceSplitter

import nltk
from llama_index.core.node_parser import SentenceWindowNodeParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import HierarchicalNodeParser

from .models import embedding

"""
Implemented two splitter methods from llama_index and 1 from langchain
TODO:
Implement the rest of the methods and update the config accordingly
For SemanticSplitterNodeParser use gemini embedding instead of openai
reference: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html#text-splitters

"""


def transform_using_SentenceSplitter(documents: list[Document], config: Dict) -> list:
    sentence_splitter = SentenceSplitter(**config)
    nodes = sentence_splitter.get_nodes_from_documents(documents)
    return nodes


def transform_using_SentenceWindowNodeParser(documents: list[Document], config: Dict) -> list:
    sentence_window_node_parser = SentenceWindowNodeParser(**config)
    nodes = sentence_window_node_parser.get_nodes_from_documents(documents)
    return nodes


def transform_using_RecursiveCharacterTextSplitter(documents: list[Document], config: Dict) -> list:
    recursive_character_text_splitter = LangchainNodeParser(
        RecursiveCharacterTextSplitter(**config))
    nodes = recursive_character_text_splitter.get_nodes_from_documents(documents)
    return nodes

def transform_using_SemanticSplitterNodeParser(documents: list[Document], config: Dict) -> list:
    '''
    config structure:

    config = {
        "embed_model": {
            "base_name": "gemini",
            "model_name": "models/embedding-001"
        }
        "node_parser": {
            "buffer_size": 1, 
            "breakpoint_percentile_threshold": 95
        }
    }
    '''
    embed_model = embedding.embed_using_GeminiEmbedding(config["embed_model"]["model_name"]) if config["embed_model"]["base_name"] == "gemini" else None
    del config["embed_model"]
    semantic_splitter_node_parser = SemanticSplitterNodeParser(
        embed_model = embed_model,
        **config["node_parser"]
    )
    nodes = semantic_splitter_node_parser.get_nodes_from_documents(documents)
    return nodes    

if __name__ == "__main__":
    pass
