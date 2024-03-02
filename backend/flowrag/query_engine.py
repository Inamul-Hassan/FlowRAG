# references
# https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html
# https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.indices.base import BaseIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import get_response_synthesizer



def query_using_RetrieverQueryEngine(index: BaseIndex, llm: LLM, config: dict) -> BaseQueryEngine:
    '''
    config = {
        "similarity_top_k": 2,
        "response_mode": "compact"
    }
    '''
    retriever = VectorIndexRetriever(index = index, similarity_top_k = config["similarity_top_k"])
    # print(retriever.retrieve("where did he complete his graduation?"))
    response_synthesizer = get_response_synthesizer(response_mode = config["response_mode"])
    query_engine = RetrieverQueryEngine.from_args(retriever = retriever, llm = llm, response_synthesizer = response_synthesizer)
    return query_engine