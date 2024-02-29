import sys
sys.path.append(r'C:\Users\vishal\Documents\AI\RAG pipeline\FlowRAG')
# sys.path.append("E:\Python\Directory\FlowRAG-main")

print(sys.path)

from backend import load
from backend import transform
from backend import index
from backend.models import embedding

from llama_index.llms.gemini import Gemini
from backend.query_engine import query_using_RetrieverQueryEngine

from dotenv import load_dotenv
load_dotenv()

documents = load.get_docs_using_SimpleDirectoryReader("storage/data")
nodes = transform.transform_using_RecursiveCharacterTextSplitter(
    documents, {"chunk_size":  4000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]})
embedding_model = embedding.embed_using_GeminiEmbedding(model_name="models/embedding-001")
vector_index = index.get_index_using_VectorStoreIndex(nodes=nodes,embedding_model=embedding_model)

# vector_index.storage_context.persist("storage/db")


from llama_index.core import Settings

Settings.llm = Gemini(model_name="models/gemini-pro")
llm = Gemini(model_name="models/gemini-pro")

# retriever = vector_index.as_retriever()
# query_engine = vector_index.as_query_engine(llm=llm)


# # response = retriever.retrieve("did pual gram ever paint?")
# retriever_response =retriever.retrieve("what is vieweb?")
query_engine = query_using_RetrieverQueryEngine(index = vector_index, llm = llm, config = {"similarity_top_k": 3, "response_mode": "compact"})
query_response =  query_engine.query("what is vieweb?")


# print(retriever_response)
print("\n\n\n\n\n")
print(query_response)