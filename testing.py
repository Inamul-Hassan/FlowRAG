listing = ["hwp", "pdf", "docx", "pptx", "ppt", "pptm", "jpg",
           "png", "jpeg", "mp3", "mp4", "csv", "epub", "md", "mbox", "ipynb", "txt", "json"]

print(listing.sort())
print(listing)
import time


from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

llm = Gemini(model_name="models/gemini-pro")
embed_model = GeminiEmbedding(model_name = "models/embedding-001")

