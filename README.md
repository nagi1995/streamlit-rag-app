
# ⚡ Streamlit RAG Application

---

# [Link](https://binginagesh.medium.com/streamlit-rag-application-5a5f2b6bc2a8) to medium blog

---

A powerful and interactive Retrieval-Augmented Generation (RAG) application built with Streamlit. This tool supports multiple document formats (PDF, TXT, DOCX), enables chunking and retrieval using advanced techniques like ChromaDB, FAISS, and pgVector, and integrates with HuggingFace LLMs to generate answers from your uploaded documents.




## ✨ Features

- **Multi-format Document Upload**: Upload PDF, TXT, or DOCX files.
- **Customizable Text Chunking**: Choose from Recursive, Character, Token, and Sentence splitters.
- **Flexible Retrieval Techniques**:
  - Vector-based (ChromaDB, FAISS, pgVector)
  - Keyword-based (BM25, TF-IDF, LSI)
  - Hybrid (Combine vector + keyword)
- **LLM Integration**: Use HuggingFace-hosted models like Mistral-7B, Phi-4-mini, and more.
- **Interactive UI**: Built using Streamlit with real-time feedback and result display.



## 🧠 How It Works

1. Upload one or more documents.
2. Choose your desired text splitting strategy and retrieval method.
3. Select your embedding and LLM model.
4. Ask a question — get contextually accurate answers with sources!



## 📁 File Structure

```
.
├── main.py                  # Main Streamlit app
├── media/
│   └── 2.mp4                # Demo video
│   └── 2.gif                # Demo GIF
├── .env                     # Environment variables for pgVector/HuggingFace
└── requirements.txt         # Python dependencies
```



## 🔧 Environment Variables

Create a `.env` file in the root directory with the following for pgVector/HuggingFace usage:

```env
PG_DBNAME=your_db
PG_USER=your_user
PG_PASSWORD=your_password
PG_HOST=your_host
PG_PORT=your_port
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```



## ▶️ Demo

https://github.com/user-attachments/assets/a8a1b31f-d222-4c72-8ab7-9cb298dea1d0

## 📦 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/nagi1995/streamlit-rag-app.git
cd streamlit-rag-app
```

2. **Create a Conda Environment**

```bash
conda create --name rag python=3.10.16 -y
conda activate rag  
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the App**

```bash
streamlit run main.py
```



## 💡 Use Cases

- Conversational document Q&A
- Chat with PDF / DOCX / TXT
- Information retrieval over large corpora
- Hybrid semantic and keyword search
- Enterprise document summarization and QA

---

## 📽 Demo Highlights

- Upload and process documents
- Choose retrieval method
- Select embedding and LLM
- Ask questions and view both answers and supporting chunks

---

## 🧑‍💻 Author

**Nagesh**  
[GitHub](https://github.com/nagi1995) | [LinkedIn](https://www.linkedin.com/in/bnagesh1/)

---

## Note: This code was generated with the assistance of ChatGPT based on instructions I provided.



