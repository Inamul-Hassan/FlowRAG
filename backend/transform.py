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


"""
Implemented two splitter methods from llama_index and 1 from langchain
TODO:
Implement the rest of the methods and update the config accordingly
For SemanticSplitterNodeParser use gemini embedding instead of openai
reference: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html#text-splitters

"""


def get_nodes_using_sentence_splitter(documents: list[Document], config: Dict) -> list:
    sentence_splitter = SentenceSplitter(**config)
    return sentence_splitter.get_nodes_from_documents(documents)


def get_nodes_using_sentence_window_node_parser(documents: list[Document], config: Dict) -> list:
    sentence_window_node_parser = SentenceWindowNodeParser(**config)
    return sentence_window_node_parser.get_nodes_from_documents(documents)


def get_nodes_using_recursive_character_text_splitter(documents: list[Document], config: Dict) -> list:
    recursive_character_text_splitter = LangchainNodeParser(
        RecursiveCharacterTextSplitter(**config))
    return recursive_character_text_splitter.get_nodes_from_documents(documents)
