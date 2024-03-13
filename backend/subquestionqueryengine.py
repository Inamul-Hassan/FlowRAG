from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

import chromadb
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding

import phoenix as px
import llama_index.core
llama_index.core.set_global_handler("arize_phoenix")
session = px.launch_app()

class SubQuestionQuerying:

    def __init__(self, data_dir: str, config: dict, llm: LLM, embed_model: BaseEmbedding):
        '''
        config = {
            "transform": {
                "chunk_size": 256
            },
            "retriever": {
                "similarity_top_k": 2,
                "num_queries": 4
            }
            "storage": {
                "db_loc": "storage/chromadb",
                "collection_name": "defaultDB"
            },
            "chat_history": {
                "loc": "storage/chat_store.json",
                "key": "user01"
            }
        }
        '''
        self.documents = SimpleDirectoryReader(input_dir = data_dir).load_data()
        self.config = config
        self.llm = llm
        self.embed_model = embed_model
    
    def store(self) -> None:
        def preprocess(documents, config: dict):
            splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(**config["transform"]))
            nodes = splitter.get_nodes_from_documents(documents)
            return nodes
        
        nodes = preprocess(documents = self.documents, config = self.config)

        # Initialize ChromaDB
        db = chromadb.PersistentClient(path = self.config["storage"]["db_loc"])
        chroma_collection = db.get_or_create_collection(name = self.config["storage"]["collection_name"])
        vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store = vector_store)

        # Store the nodes to DB and create index out of it
        VectorStoreIndex(
            nodes = nodes, 
            embed_model = self.embed_model, 
            storage_context = storage_context
        )

        # Store chat history
        chat_store = SimpleChatStore()
        chat_store.add_message(self.config["chat_history"]["key"], message= ChatMessage(
                    role=MessageRole.USER,
                    content="Hello assistant, we are having a insightful discussion about Paul Graham today."))
        chat_store.add_message(self.config["chat_history"]["key"], message= ChatMessage(
                    role=MessageRole.ASSISTANT, 
                    content="Okay, sounds good."))
        chat_store.persist(persist_path=self.config["chat_history"]["loc"])   

    def query(self, query_str: str, debug: bool = False) -> str:

        def load(db_loc: str, collection_name: str, chat_history_loc: str):
            db = chromadb.PersistentClient(path = db_loc)
            chroma_collection = db.get_or_create_collection(name = collection_name)
            vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store = vector_store,
                embed_model = self.embed_model
            )

            chat_store = SimpleChatStore.from_persist_path(chat_history_loc)
            
            return index, chat_store

        # Load index from storage and chat history
        index, chat_store = load(
            db_loc = self.config["storage"]["db_loc"], 
            collection_name = self.config["storage"]["collection_name"], 
            chat_history_loc = self.config["chat_history"]["loc"]
        )

        retriever = VectorIndexRetriever(
            index = index, 
            similarity_top_k = self.config["retriever"]["similarity_top_k"]
        ) 

        # Query engine
        vector_query_engine = RetrieverQueryEngine(retriever = retriever)

        query_engine_tools = [
            QueryEngineTool(
                query_engine = vector_query_engine,
                metadata = ToolMetadata(
                    name="paul_graham_essay",
                    description="Paul Graham essay on What I Worked On"
                )
            ),   
        ]
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools = query_engine_tools
        )

        # Setting system prompt to consider chat hostory
        custom_prompt = PromptTemplate("""Given a conversation (between Human and Assistant) and a follow up message from Human, rewrite the message to be a standalone question that captures all relevant context from the conversation and that standalone question can be used to query a vector database to get the relavent data.\n<Chat History>\n{chat_history}\n<Follow Up Message>\n{question}\n<Standalone question>""")
        # Pull chat history for the provided key
        custom_chat_history = chat_store.get_messages(self.config["chat_history"]["key"]) 

        # Initialize chat engine
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine = query_engine,
            condense_question_prompt = custom_prompt,
            chat_history = custom_chat_history,
            llm = self.llm
        )
        
        response = chat_engine.chat(query_str).response

        # Appending current chat query and its response
        chat_store.add_message(self.config["chat_history"]["key"], ChatMessage(role=MessageRole.USER, content = query_str))
        chat_store.add_message(self.config["chat_history"]["key"], ChatMessage(role=MessageRole.ASSISTANT, content = response))
        chat_store.persist(persist_path="storage/chat_store.json")

        if debug:
            import time
            while True:
                time.sleep(100)

        return response

if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    llm = Gemini(model_name="models/gemini-pro")
    embed_model = GeminiEmbedding(model_name = "models/embedding-001")
    Settings.embed_model = embed_model
    Settings.llm = llm

    rag = SubQuestionQuerying(
        data_dir = "storage/data",
        config = {
            "transform": {
                "chunk_size": 512
            },
            "retriever": {
                "similarity_top_k": 5,
                "num_queries": 4
            },
            "storage": {
                "db_loc": "storage/chromadb",
                "collection_name": "defaultDB"
            },
            "chat_history": {
                "loc": "storage/chat_store.json",
                "key": "user01"
            }
        },
        llm = llm,
        embed_model = embed_model
    )
    rag.store()

    print(rag.query("what made the author interested in the AI?", debug = False))
    print(rag.query("Eloborate more", debug = True))