from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

llm = ChatGroq(model="opeani/oss-gpt-120b")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = SentenceTransformer(model_name)

# sentences = ["This is an example sentence.", "Each sentence is converted into a vector."]

# embedding = model.encode(sentences)
# print(embedding[0][:10])

from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore(embedding)
