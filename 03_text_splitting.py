# https://python.langchain.com/docs/tutorials/retrievers/

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

all_splits = text_splitter.split_documents(docs)

# print(len(chunks))
# print(all_splits)

# for i, chunk in enumerate(chunks, 1):
#     print(f"Chunk {i}: {chunk}")

# -------Generating embeddings (convert text into embeddings)-------

from langchain_huggingface import HuggingFaceEmbeddings

# Use HuggingFace embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# generate embedding 
# text = "Artificial Intelligence is transforming the world."
# vector = embedding_model.embed_query(text)

# print(f"Embedding langth : {len(vector)}")
# print(vector[:10])

vector_1 = embedding.embed_query(all_splits[0].page_content)
vector_2 = embedding.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

# --------- store embbeddings in vector database ----------
# https://python.langchain.com/docs/tutorials/retrievers/
# Vector stores