#chatbot.py
#→ User types a question → retrieve top chunks → build a prompt with “Knowledge” → send to Hugging Face model → show answer in Gradio UI.

import os
from dotenv import load_dotenv
import gradio as gr

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import FAISS

# Load env vars
load_dotenv()

# configuration
DATA_PATH = "data"
FAISS_PATH = "faiss_index"

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "HuggingFaceH4/zephyr-7b-beta")

# 1. Embeddings (must be same as in ingest_database.py)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load FAISS vector store
vector_store = FAISS.load_local(
    FAISS_PATH,
    embeddings_model,
    allow_dangerous_deserialization=True,  # safe if it's your own index
)

# Set up retriever (similar to original code)
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Hugging Face chat model (Inference API)
llm_endpoint = HuggingFaceEndpoint(
    repo_id=HF_MODEL_ID,
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    huggingfacehub_api_token=HF_TOKEN,
)

chat_model = ChatHuggingFace(llm=llm_endpoint)


# 4. This function is called for every user message
def answer(message, history):
    # 1. Retrieve relevant chunks from FAISS
    docs = retriever.invoke(message)

    # 2. Join them into a single "knowledge" text
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    # 3. Simple, strict RAG prompt
    rag_prompt = f"""
You are an assistant that answers questions strictly based on the provided knowledge.

Rules:
- Use ONLY the text in the "Knowledge" section.
- If the answer is not clearly in the knowledge, say: "I don't know from the given notes."
- Answer in simple, clear English, like a human tutor.
- Keep the answer short (4–6 sentences).

Knowledge:
{knowledge}

Question: {message}

Answer:
""".strip()

    # 4. Call the Hugging Face chat model
    response = chat_model.invoke(rag_prompt)

    # 5. Return only the text answer
    return response.content.strip()


# 5. Gradio ChatInterface (almost same as tutorial)
chat_ui = gr.ChatInterface(
    fn=answer,
    textbox=gr.Textbox(
        placeholder="Ask something about the PDF...",
        container=False,
        autoscroll=True,
        scale=7,
    ),
)

if __name__ == "__main__":
    chat_ui.launch(
        server_name="127.0.0.1",  
        server_port=7860        
    )

