from typing import Dict
from llama_index.core.readers import Document

from llama_index.core.node_parser import SentenceSplitter

import nltk
from llama_index.core.node_parser import SentenceWindowNodeParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.base.embeddings.base import BaseEmbedding

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import HierarchicalNodeParser


"""
Implemented two splitter methods from llama_index and 1 from langchain
TODO:
Implement the rest of the methods and update the config accordingly
For SemanticSplitterNodeParser use gemini embedding instead of openai
reference: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html#text-splitters

"""


def get_nodes_using_SentenceSplitter(documents: list[Document], config: Dict) -> list:
    sentence_splitter = SentenceSplitter(**config)
    nodes = sentence_splitter.get_nodes_from_documents(documents)
    return nodes


def get_nodes_using_SentenceWindowNodeParser(documents: list[Document], config: Dict) -> list:
    sentence_window_node_parser = SentenceWindowNodeParser(**config)
    nodes = sentence_window_node_parser.get_nodes_from_documents(documents)
    return nodes


def get_nodes_using_RecursiveCharacterTextSplitter(documents: list[Document], config: Dict) -> list:
    recursive_character_text_splitter = LangchainNodeParser(
        RecursiveCharacterTextSplitter(**config))
    nodes = recursive_character_text_splitter.get_nodes_from_documents(
        documents)
    return nodes


def get_nodes_using_SemanticSplitterNodeParser(documents: list[Document], config: Dict) -> list:
    embedding = BaseEmbedding(model_name="gemini-pro",
                              embed_batch_size=config["embed_batch_size"])
    semantic_splitter_node_parser = SemanticSplitterNodeParser(
        embed_model=embedding, **config)
    nodes = semantic_splitter_node_parser.build_semantic_nodes_from_documents(
        documents)
    return nodes


if __name__ == "__main__":
    pass
