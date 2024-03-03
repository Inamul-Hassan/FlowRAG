import sys
sys.path.append(r'C:\Users\vishal\Documents\AI\RAG pipeline\FlowRAG')

from backend.reciprocalrerankfusionretriever import ReciprocalRerankFusionRetriever

from backend.flowrag.models.embedding import embed_using_GeminiEmbedding
from backend.flowrag.models.llm import chat_using_Gemini

from llama_index.core import Settings


llm = chat_using_Gemini()
embed_model = embed_using_GeminiEmbedding()
Settings.llm = llm
query_str = "Who are all the people he asked to be the president?"

response = ReciprocalRerankFusionRetriever(
    data_dir="storage/data",
    config = {
            "transform": {
                "chunk_size": 256
            },
            "retriever": {
                "similarity_top_k": 2,
                "num_queries": 4
            }
        },
    llm = llm,
    embed_model = embed_model
    ).query(query_str = query_str)

print(f"User query: {query_str}\nResponse: {response}")
