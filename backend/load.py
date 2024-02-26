from llama_index.core import SimpleDirectoryReader

SUPPORTED_EXTENSIONS = ["hwp", "pdf", "docx", "pptx", "ppt", "pptm", "jpg",
                        "png", "jpeg", "mp3", "mp4", "csv", "epub", "md", "mbox", "ipynb", "txt", "json"]

# validating user input
LOADER_EXTENSTIONS_MAPPING = {
    "SimpleDirectoryReader": [".hwp", ".pdf", ".docx", ".pptx", ".ppt", ".pptm", ".jpg",
                              ".png", ".jpeg", ".mp3", ".mp4", ".csv", ".epub", ".md", ".mbox", ".ipynb", ".txt", ".json"]
}


def validate_input_file(input_file: str) -> str:
    try:
        extenstion = input_file.split(".")[-1]
    except Exception as e:
        raise ValueError(f"Invalid input file: {e}")
    if extenstion.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extenstion}")
    else:
        return extenstion.lower()


# Load
"""
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
        ".txt": TextReader,
"""


reader = SimpleDirectoryReader(
    input_dir="storage/data",
)

docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

print(type(docs[0]))
