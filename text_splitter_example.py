from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
These machines are programmed to think and learn like humans.
AI applications include natural language processing, computer vision, robotics, and more.
"""

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,      # max 50 chars
    chunk_overlap=10,   # 10 chars overlap
)

# Split text
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")
