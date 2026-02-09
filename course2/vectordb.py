import pdfplumber
import tiktoken 
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text 
raw_text = extract_text("hitler.pdf")

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0

    while i<len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i+= chunk_size - overlap

    return chunks
chunks = chunk_text(raw_text)

encoder = tiktoken.get_encoding("cl100k_base")

def tokenize(text):
    return encoder.encode(text)

tokens = tokenize(chunks[0])

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MineLM-L6-v2")
embedding = model.encode(chunks).tolist()

from pinecone import Pinecone
pc = Pinecone(api_key="PINECONE_API_KEY")
index = pc.Index("pdf-index")

vectors = []
for i, emb in enumerate(embedding):
    vectors,append({
        "id": f"chunk-{i}",
        "values":emb,
        "metadata": {
            "text": chunks[i],
            "source": "hitler.pdf"
        }
    })
index.upsert(vectors=vectors)

query_text = "explain convolutional neural networks"

query_embedding = model.encode(query_text).tolist()

result = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

for match in result["matches"]:
    print(match["metadata"]["text"])

from transformers import pipline
llm = pipline(
     "text2text-generation",
     model="google/flan-t5-base"
 )    
context = " ".join(
    [m["metadata"]["text"] for m in result["matches"]] 
)

prompt = f"""
answer the question using the context below. 

Context:
{context}

Question:
{query_text}
"""

response = llm(prompt, max_length=256)
print(response[0]["generated_text"])