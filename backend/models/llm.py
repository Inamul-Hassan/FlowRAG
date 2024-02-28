from llama_index.llms.gemini import Gemini

def chat_using_Gemini(config: dict = {"model_name": "models/gemini-pro"}) -> Gemini:
    llm = Gemini(**config)
    return llm