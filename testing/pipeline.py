import sys
sys.path.append(r'C:\Users\vishal\Documents\AI\RAG pipeline\FlowRAG')

print(sys.path)

from backend import load
from backend import transform
from backend import index

from llama_index.llms.gemini import Gemini


documents = load.get_docs_using_SimpleDirectoryReader("storage/data")
nodes = transform.get_nodes_using_recursive_character_text_splitter(
    documents, {"chunk_size":  4000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]})
embedding_model = index.get_embedding_using_GeminiEmbedding()
vector_index = index.get_vector_index_using_VectorStoreIndex(nodes=nodes,embedding_model=embedding_model)

llm = Gemini(model_name="models/gemini-pro")

retriever = vector_index.as_retriever()
query_engine = vector_index.as_query_engine(llm=llm)


# response = retriever.retrieve("did pual gram ever paint?")
retriever_response =retriever.retrieve("what is vieweb?")
query_response =  query_engine.query("what is vieweb?")


print(retriever_response)
print("\n\n\n\n\n")
print(query_response)