import streamlit as st
import os
import shutil
import datetime
import tempfile
import concurrent.futures
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma, FAISS, PGVector
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import faiss
import psycopg2
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import fasttext

# Load environment variables
load_dotenv()

@st.cache_resource
def get_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource
def get_llm_model(repo_id, temperature, max_tokens, top_k, top_p):
    return HuggingFaceHub(repo_id=repo_id, model_kwargs={
        "temperature": temperature, "max_tokens": max_tokens, "top_k": top_k, "top_p": top_p
    })

def get_embedding_size(embeddings):
    """Gets the embedding size dynamically."""
    return len(embeddings.embed_query("test"))  # Assuming `embed_query` returns a list

def get_timestamp():
    """Generates a timestamp in the format YYYYMMDD_HHMMSS."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_persist_directory(base_dir, embedding_size):
    """Creates a new directory structure: ./db/<embedding-size>/YYYYMMDD_HHMMSS/"""
    timestamp = get_timestamp()
    new_dir = os.path.join(base_dir, str(embedding_size), timestamp)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

def get_pgvector_table_name(embedding_size):
    """Generates a unique table name: documents_<embedding-size>_YYYYMMDD_HHMMSS"""
    timestamp = get_timestamp()
    return f"documents_{embedding_size}_{timestamp}"

def get_vector_store(vector_store, docs, embeddings, base_dir="./db"):
    embedding_size = get_embedding_size(embeddings)  # Get embedding dimension

    if vector_store == "ChromaDB":
        persist_directory = get_persist_directory(base_dir, embedding_size)
        return Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

    elif vector_store == "FAISS":
        persist_directory = get_persist_directory(base_dir, embedding_size)
        faiss_store = FAISS.from_documents(docs, embeddings)
        faiss_store.save_local(persist_directory)
        return faiss_store

    elif vector_store == "pgVector":
        table_name = get_pgvector_table_name(embedding_size)  # Unique table name
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DBNAME"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR({embedding_size})
            )
        """)
        conn.commit()
        
        vectors = embeddings.embed_documents([d.page_content for d in docs])
        cur.executemany(f"INSERT INTO {table_name} (content, embedding) VALUES (%s, %s)", 
                        [(doc.page_content, vec) for doc, vec in zip(docs, vectors)])
        conn.commit()
        cur.close()
        conn.close()
        
        connection_string = f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DBNAME')}"
        return PGVector(collection_name=table_name, embedding_function=embeddings, connection_string=connection_string)


def keyword_retriever(docs, method):
    texts = [doc.page_content for doc in docs]
    
    if method == "BM25":
        tokenized_corpus = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return lambda query: bm25.get_top_n(query.split(), docs, n=5)
    
    elif method == "TF-IDF":
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return lambda query: [docs[i] for i in tfidf_matrix.dot(vectorizer.transform([query]).T).toarray().flatten().argsort()[-5:][::-1]]
    
    elif method == "LSI":
        dictionary = corpora.Dictionary([text.split() for text in texts])
        corpus = [dictionary.doc2bow(text.split()) for text in texts]
        lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
        return lambda query: [docs[i] for i in sorted(range(len(texts)), key=lambda i: sum(x[1] for x in lsi_model[dictionary.doc2bow(query.split())]) if i < len(texts) else 0, reverse=True)[:5]]
    
    

QA_CHAIN_PROMPT = PromptTemplate.from_template(
    """Use the following retrieved context to answer the question.
    If you don't know, say "I don't know." Keep it concise.

    {context}
    Question: {question}
    Answer:"""
)

st.set_page_config(page_title="Streamlit RAG App", layout="wide")
st.title("⚡ Streamlit RAG Application")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🛠️ Retrieval Settings")
    retrieval_type = st.selectbox("Select Retrieval Type", ["Hybrid", "Vector-based", "Keyword-based"])

    if retrieval_type in ["Vector-based", "Hybrid"]:
        vector_store = st.selectbox("Select Vector Store", ["ChromaDB", "FAISS", "pgVector"])
        embedding_model = st.selectbox("Select Embedding Model", [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-Mpnet-base-v2"
        ])

    if retrieval_type in ["Keyword-based", "Hybrid"]:
        keyword_method = st.selectbox("Select Keyword Retrieval Method", ["BM25", "TF-IDF", "LSI"])

    st.markdown("### 🤖 LLM Settings")
    llm_model = st.selectbox("Select LLM", ["mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-7b", "stabilityai/stablelm-tuned-alpha-3b"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    max_tokens = st.slider("Max Tokens", 50, 4096, 4000)
    top_k = st.slider("Top-K", 1, 50, 40)
    top_p = st.slider("Top-P", 0.0, 1.0, 0.9)

with col2:
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt", "docx"])

    if st.button("Process Files"):
        docs = []
        with st.spinner("Processing files..."):
            def process_file(file):
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                loader = PyPDFLoader(temp_file.name) if file.name.endswith(".pdf") else TextLoader(temp_file.name) if file.name.endswith(".txt") else Docx2txtLoader(temp_file.name)
                return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(loader.load())

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for result in executor.map(process_file, uploaded_files):
                    docs.extend(result)
        

        embeddings = get_embedding_model(embedding_model) if retrieval_type in ["Vector-based", "Hybrid"] else None
        st.session_state.retriever = (
            get_vector_store(vector_store, docs, embeddings).as_retriever(search_kwargs={"k": 5})
            if retrieval_type == "Vector-based" else keyword_retriever(docs, keyword_method)
            if retrieval_type == "Keyword-based" else (get_vector_store(vector_store, docs, embeddings).as_retriever(search_kwargs={"k": 5}), keyword_retriever(docs, keyword_method))
        )

    query = st.text_input("Enter your query")
    if query and "retriever" in st.session_state:
        retrieved_chunks = []
        if retrieval_type == "Keyword-based":
            retrieved_chunks = st.session_state.retriever(query)
        elif retrieval_type == "Vector-based":
            retrieved_chunks = st.session_state.retriever.get_relevant_documents(query)
        elif retrieval_type == "Hybrid":
            vector_retriever, keyword_retriever_func = st.session_state.retriever
            vector_results = vector_retriever.get_relevant_documents(query)
            keyword_results = keyword_retriever_func(query)

            # Merge without duplicates while maintaining order
            seen_texts = set()
            for doc in vector_results + keyword_results:
                if doc.page_content not in seen_texts:
                    seen_texts.add(doc.page_content)
                    retrieved_chunks.append(doc)

        llm = get_llm_model(llm_model, temperature, max_tokens, top_k, top_p)
        response = llm.invoke(QA_CHAIN_PROMPT.format(context="\n".join([chunk.page_content for chunk in retrieved_chunks]), question=query))
        
        final_answer = response.split("Answer:")[-1].strip()
        st.markdown("### 💡 Answer")
        st.write(final_answer)  # Only printing the answer

        # Display Retrieved Chunks
        st.markdown("### 🔍 Retrieved Chunks")
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}:**\n> {chunk.page_content}")

