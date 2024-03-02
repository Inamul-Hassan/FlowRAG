from flowrag.load import get_docs_using_SimpleDirectoryReader



docs = get_docs_using_SimpleDirectoryReader("storage/data")

print(docs)