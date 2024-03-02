import sys
# sys.path.append(r'C:\Users\vishal\Documents\AI\RAG pipeline\FlowRAG')
sys.path.append("E:\Python\Directory\FlowRAG-main")

from backend import load
from backend import transform
from backend import index
from backend.models import embedding
from llama_index.llms.gemini import Gemini
from backend.query_engine import query_using_RetrieverQueryEngine
from llama_index.core import Settings

from dotenv import load_dotenv
load_dotenv()

# Load the models
embedding_model = embedding.embed_using_GeminiEmbedding(model_name="models/embedding-001")
llm = Gemini(model_name="models/gemini-pro")
Settings.llm = Gemini(model_name="models/gemini-pro")

# Load the documents
documents = load.get_docs_using_SimpleDirectoryReader("storage/data")

# Transform the documents
nodes = transform.transform_using_RecursiveCharacterTextSplitter(
    documents, {"chunk_size":  4000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]})

# Index the documents
vector_index = index.get_index_using_VectorStoreIndex(nodes=nodes,embedding_model=embedding_model)

# Persist the index
# vector_index.storage_context.persist("storage/db")

# Query the index
query_engine = query_using_RetrieverQueryEngine(index = vector_index, llm = llm, config = {"similarity_top_k": 3, "response_mode": "refine"})
query_response =  query_engine.query("how many grad school does the author apply to?")


# print(retriever_response)
print("\n\n\n\n\n")
print(query_response)