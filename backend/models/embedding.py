from llama_index.embeddings.gemini import GeminiEmbedding
def embed_using_GeminiEmbedding(model_name: str = "models/embedding-001", **kwargs) -> GeminiEmbedding:
    '''
    models - [models/embedding-001, ]
    default_model - models/embedding-001
    '''
    embed_model = GeminiEmbedding(model_name = model_name, **kwargs)
    return embed_model

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
def embed_using_HuggingFaceEmbedding(model_name: str ="BAAI/bge-small-en-v1.5", **kwargs) -> HuggingFaceEmbedding:
    '''
    models - [BAAI/bge-small-en-v1.5, ]
    default_model - BAAI/bge-small-en-v1.5
    '''
    embed_model = HuggingFaceEmbedding(model_name = model_name, **kwargs)
    return embed_model


from llama_index.embeddings.openai import OpenAIEmbedding
def embed_using_OpenAIEmbedding(model_name: str = "text-embedding-ada-002", **kwargs) -> OpenAIEmbedding:
    '''
    models - [davinci, curie, babbage, ada, text-embedding-ada-002, text-embedding-3-large text-embedding-3-small, ]
    default_model - text-embedding-ada-002
    '''
    embed_model = OpenAIEmbedding(model = model_name, **kwargs)
    return embed_model