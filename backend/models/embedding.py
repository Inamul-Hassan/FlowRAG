from llama_index.embeddings.gemini import GeminiEmbedding

def embed_using_GeminiEmbedding(model_name: str) -> GeminiEmbedding:
    embedding_model = GeminiEmbedding(model_name = model_name)
    return embedding_model