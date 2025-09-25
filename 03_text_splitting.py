from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

file_path = "./nke-10k-2023.pdf"
# file_path = "./personal_data.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()
# print(len(docs))

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

# ------------- Split text into chunks -----------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

chunks = text_splitter.split_documents(docs)

print(len(chunks))
# print(all_splits)

# for i, chunk in enumerate(chunks, 1):
#     print(f"Chunk {i}: {chunk}")

# -------Generating embeddings (convert text into embeddings)-------
