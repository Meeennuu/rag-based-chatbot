**RAG-Based Chatbot using Hugging Face + FAISS + Gradio**

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly based on the content of PDF documents. It uses:

-FAISS for vector search
-Sentence Transformers for embeddings
-Hugging Face LLM models for natural language generation
-Gradio for an interactive chat UI

This chatbot guarantees accurate, context-specific answers and avoids hallucinations.

**Features**

Upload & process PDF documents
Automatic text chunking using RecursiveCharacterTextSplitter
Embeddings generated using MiniLM (all-MiniLM-L6-v2)
Fast retrieval using FAISS vector database
LLM-powered responses using Hugging Face models
Clean and simple web UI via Gradio
Runs completely on your local machine
Option to host on your local network

**📁 Project Structure**
rag-based-chatbot/

├── chatbot.py

├── ingest_database.py  

├── requirements.txt  

├── NIPS-2017-attention-is-all-you-need-Paper.pdf 

└── (FAISS index is created after ingestion)

**🛠 Installation & Setup**

1️⃣ Clone the repository
git clone https://github.com/Meeennuu/rag-based-chatbot.git
cd rag-based-chatbot

2️⃣ Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

**🔑 Environment Variables**

Create a .env file in the project root:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
HF_MODEL_ID=HuggingFaceH4/zephyr-7b-beta

You can generate a token here:
👉 https://huggingface.co/settings/tokens

**🧠 Build the FAISS Vector Database**

Run this script after adding or modifying PDFs:
python ingest_database.py

This will create a folder named faiss_index/.

**💬 Run the Chatbot**
python chatbot.py

You will see:
Running on local URL: http://127.0.0.1:7860/

Open it in your browser to use the chatbot.
