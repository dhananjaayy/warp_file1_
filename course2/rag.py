#extracting data
import pdfplumber
import chromadb
import openai
def extract_text_from_pdf(pdf_path):
    full_text=""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text+=page.extract_text() +"\n"
    return full_text
 #break the text        
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start=0

    while start<lem(text):
       end=start + chunk_size
       chunk=text[start:end]
       chunks.append(chunk)
       start+=chunk_size - overlap

    return chunks
# ebmading
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniM-L6-v2")
def embed_chunks(chunks):
    return model.encode(chunks)

#storing data in vectors
client  = chromadb.Client()
collection = client.create_collection("book_collection")
def store_chunks(chunks, embeddings):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
        )    
#finding revelant info
def retrieve(query, k=3):
    query_embedding=model.encode([query])
    result=collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )        
    return results["documents"][0]
    context="\n\n".join(retrieved_docs)

#memory handiling
chat_history_append({
    "role": "user",
    "countent": user_input
})

def summarize_chat(history, llm):
    conversation="\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in history]
    
    )
                    
    prompt = f"summarize this conversation:\n{conversation}"
    summary = llm(prompt)
    return summary
chat_summary= ""
#prompt engineering
def build_prompt(user_query, retrieved_docs, chat_history, chat_summary):
    history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    )
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
you are a helpful AI assistant.

Use the following retrieved context from the book to answer the question.
If the answer is not in the context, say you don't know. 
chat_summary:
{chat_summary}
chat history:
{chat_history}
retrieved context
{context}
user question:
{user_query}

answer:
"""
    return prompt

#call llm
from openai import OpenAI
client=OpenAI()
def call_llm(prompt):
    response = client.Chat.completion.create(
        model = "gpt-4o-mini",
        message=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choice"][0]["message"]["content"]
#putting all together
def rag_agent(user_query):
    retrieved_docs = retrieve(user_query)
    prompt = build_prompt(
        user_query,
        retrieved_docs,
        chat_history,
        chat_summary
    )
    answer=call_llm(prompt)

    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer

import streamlit as st
st.title("RAG Book Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("ask a question")

if user_input:
    answer = rag_agent(user_input)

    st.session_state.chat_history.append(("you", user_input))
    st.session_state.chat_history.append(("Bot", answer))

for role, msg in st.session_state.chat_history:
    st.write(f"**{role}:** {msg}")
