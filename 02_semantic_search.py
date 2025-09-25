from langchain_community.document_loaders import PyPDFLoader

file_path = "./nke-10k-2023.pdf"
# file_path = "./personal_data.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()
# print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)