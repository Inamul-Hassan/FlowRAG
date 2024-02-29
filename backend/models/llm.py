from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI

def chat_using_Gemini(config: dict = {"model_name": "models/gemini-pro"}) -> Gemini:
    llm = Gemini(**config)
    return llm

def chat_using_OpenAI(config: dict = {"model": "gpt-3.5-turbo"}) -> OpenAI:
    llm = OpenAI(**config)
    return llm

def chat_using_Ollama(config: dict = {"model": "llama-2"}) -> Ollama:
    llm = Ollama(**config)
    return llm

def chat_using_Anthropic(config: dict = {"model": "claude-2"}) -> Anthropic:
    llm = Anthropic(**config)
    return llm

def chat_using_MistralAI(config: dict = {"model": "mistral-tiny"}) -> MistralAI:
    llm = MistralAI(**config)
    return llm