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
from backend.flowrag.util import log_time

# Load the models
embedding_model = embedding.embed_using_GeminiEmbedding(model_name="models/embedding-001")
llm = Gemini(model_name="models/gemini-pro")
Settings.llm = Gemini(model_name="models/gemini-pro")

@log_time
def pipeline_04(mode:str,query:str)->str:
  """
  mode: create or query
  query: the query to be searched
  """
  match mode:
    case "create":
      # Load the documents
      documents = load.get_docs_using_SimpleDirectoryReader("storage/data")

      # Transform the documents
      nodes = transform.transform_using_RecursiveCharacterTextSplitter(
          documents, {"chunk_size":  4000,
                      "chunk_overlap": 200,
                      "separators": ["\n\n", "\n", " ", ""]})

      # Database configuration
      storage_context = index.save_to_qdrant(collection_name="PaulGramEassay01",isLocal=True,config=None)

      # Index the documents
      vector_index = index.get_index_using_VectorStoreIndex(nodes=nodes,embedding_model=embedding_model,storage_context=storage_context)
    case "query":
      # Get indexed documents from local storage
      vector_index = index.load_from_qdrant("PaulGramEassay01",isLocal=True,embedding_model=embedding_model)
    case _:
      raise ("Invalid mode") 

  # Query the index
  query_engine = query_using_RetrieverQueryEngine(index = vector_index, llm = llm, config = {"similarity_top_k": 5, "response_mode": "compact"})
  query_response =  query_engine.query(query)

  # print(retriever_response)
  print("\n\n\n\n\n")
  print(query_response)
  return query_response

if __name__ == "__main__":
  pipeline_04("query",query="what is vieweb?")