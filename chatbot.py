import os
from dotenv import load_dotenv
import gradio as gr

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

# Embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# Build VectorStore from PDFs
# -------------------------
def build_vectorstore(files):
    docs = []

    print("📄 Received Files:", files)

    for f in files:
        print("➡ Loading:", f)
        loader = PyPDFLoader(f)  # IMPORTANT FIX
        pages = loader.load()
        docs.extend(pages)

    print("📄 Total pages loaded:", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)
    print("🔍 Chunks created:", len(chunks))

    vs = FAISS.from_documents(chunks, embeddings_model)
    return vs


# -------------------------
# LLM Endpoint (HuggingFace)
# -------------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id=HF_MODEL_ID,
    task="text-generation",
    max_new_tokens=300,
    temperature=0.1,
    do_sample=False,
    huggingfacehub_api_token=HF_TOKEN,
)

chat_model = ChatHuggingFace(llm=llm_endpoint)


# -------------------------
# RAG Answer Function
# -------------------------
def answer_question(question, vs):
    print("🔎 Query:", question)

    docs = vs.similarity_search(question, k=3)
    print("📚 Retrieved docs:", len(docs))

    knowledge = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an expert assistant that answers questions ONLY using the information inside <knowledge>.  

Your goal is to produce a **highly detailed, well-structured, multi-paragraph explanation** that stays 100% faithful to the source text.

Rules:
1. Use ONLY the knowledge provided. Do NOT add outside information.  
2. If the answer is NOT found in the knowledge, reply:  
   "The document does not contain this information."  
3. When the answer exists, produce:  
   • A clear overview  
   • Deep explanation  
   • Step-by-step details  
   • Important concepts and definitions  
   • Examples, if found  
4. Write in a **rich, detailed, student-friendly style**.

<knowledge>
{knowledge}
</knowledge>

Question: {question}

Now produce a detailed answer:
"""

    # ChatHuggingFace requires messages format
    result = chat_model.invoke([
        {"role": "user", "content": prompt}
    ])

    return result.content.strip()

# -------------------------
# Build Knowledge Base
# -------------------------
def build_knowledge(files):
    if not files:
        return None, "⚠ Please upload at least one PDF."

    vs = build_vectorstore(files)
    return vs, "✅ Knowledge Base Created Successfully!"


# -------------------------
# Chat Function
# -------------------------
def chat_fn(message, history, vs):
    print("💬 Chat called.")

    if vs is None:
        history.append((message, "❌ No knowledge base found. Upload PDFs and click Build."))
        return history, ""

    answer = answer_question(message, vs)

    # GRADIO CHATBOT FORMAT → list of (user, bot) tuples
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, ""


# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("## 📚 RAG Chatbot — Upload PDFs and Ask Questions")

    files = gr.File(file_count="multiple", file_types=[".pdf"])
    build_btn = gr.Button("Build Knowledge Base")

    status = gr.Markdown()
    vs_state = gr.State(None)

    chatbot = gr.Chatbot()
    msg_box = gr.Textbox(placeholder="Ask something...")

    # Build vectorstore
    build_btn.click(
        build_knowledge,
        inputs=[files],
        outputs=[vs_state, status],
    )

    # Chat callback
    msg_box.submit(
        chat_fn,
        inputs=[msg_box, chatbot, vs_state],
        outputs=[chatbot, msg_box],
    )


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)

