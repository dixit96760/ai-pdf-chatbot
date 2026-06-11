from pypdf import PdfReader
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv
import os

# ------------------------
# Load Environment Variables
# ------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

# ------------------------
# Read PDF
# ------------------------
reader = PdfReader("BEE_42_Questions_Exam_Notes.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

print("PDF Loaded Successfully!")
print("Total Characters:", len(text))

# ------------------------
# Chunking
# ------------------------
chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

print("Total Chunks:", len(chunks))

# ------------------------
# Create ChromaDB Collection
# ------------------------
client = chromadb.Client()

try:
    collection = client.get_collection("bee_notes")
except:
    collection = client.create_collection("bee_notes")

# ------------------------
# Store Embeddings
# ------------------------
for i, chunk in enumerate(chunks):

    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=chunk
    )

    embedding = result["embedding"]

    try:
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk]
        )
    except:
        pass

print("Embeddings Stored Successfully!")

# ------------------------
# User Question
# ------------------------
question = input("\nAsk a question: ")

# Convert question to embedding
question_embedding = genai.embed_content(
    model="models/gemini-embedding-001",
    content=question
)["embedding"]

# ------------------------
# Search Vector Database
# ------------------------
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# Combine top chunks
retrieved_chunks = "\n\n".join(results["documents"][0])

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{retrieved_chunks}

Question:
{question}

Answer:
"""

# ------------------------
# Generate Answer
# ------------------------
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(prompt)

print("\n========================")
print("RAG ANSWER")
print("========================\n")

print(response.text)