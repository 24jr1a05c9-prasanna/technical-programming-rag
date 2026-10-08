# ============================================================
# COLLEGE PROGRAMMING RAG APPLICATION
# Streamlit + LangChain + HuggingFace + Chroma + Gemini
# ============================================================

import os
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Technical Programming RAG",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎓 College Programming RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about programming languages using Retrieval-Augmented Generation</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 RAG Information")

    st.write("""
    This application uses:

    • LangChain  
    • HuggingFace Embeddings  
    • ChromaDB  
    • Google Gemini  
    • Retrieval-Augmented Generation
    """)

    st.divider()

    st.subheader("📖 Languages Covered")

    st.write("""
    • C  
    • C++  
    • Java  
    • Python  
    • JavaScript  
    • C#  
    • Go  
    • Rust  
    • Kotlin  
    • Swift
    """)

    st.divider()

    st.subheader("⚙️ RAG Pipeline")

    st.write("""
    Document
    ↓
    Text Splitting
    ↓
    Embeddings
    ↓
    ChromaDB
    ↓
    Similarity Search
    ↓
    Top-K Retrieval
    ↓
    Context
    ↓
    Gemini
    ↓
    Answer
    """)


# ============================================================
# GEMINI API KEY
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error("❌ GEMINI_API_KEY is not configured.")

    st.info("""
    Set your Gemini API key in PowerShell before starting Streamlit.

    Example:

    $env:GEMINI_API_KEY="YOUR_API_KEY"
    """)

    st.stop()


# ============================================================
# LOAD DOCUMENT
# ============================================================

@st.cache_resource
def load_document():

    loader = TextLoader(
        "college.txt",
        encoding="utf-8"
    )

    documents = loader.load()

    return documents


# ============================================================
# TEXT SPLITTING
# ============================================================

@st.cache_resource
def create_chunks():

    documents = load_document()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

@st.cache_resource
def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ============================================================
# CREATE CHROMA VECTOR STORE
# ============================================================

@st.cache_resource
def create_vectorstore():

    chunks = create_chunks()

    embeddings = create_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="college_programming_rag"
    )

    return vectorstore


# ============================================================
# LOAD GEMINI
# ============================================================

@st.cache_resource
def create_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key
    )

    return llm


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

with st.spinner("🔄 Loading RAG system..."):

    try:

        documents = load_document()

        chunks = create_chunks()

        vectorstore = create_vectorstore()

        llm = create_llm()

    except Exception as e:

        st.error("❌ Error while loading the RAG system.")

        st.exception(e)

        st.stop()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success("✅ RAG system loaded successfully!")


# ============================================================
# PROJECT STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Documents",
        len(documents)
    )

with col2:

    st.metric(
        "Chunks",
        len(chunks)
    )

with col3:

    st.metric(
        "Top-K",
        2
    )

with col4:

    st.metric(
        "Embedding",
        "MiniLM"
    )


st.divider()


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💡 Example Questions")

example_questions = [
    "What is Python used for?",
    "What is Java?",
    "Which language is used for Android development?",
    "What are the advantages of C?",
    "What is JavaScript used for?",
    "What is Rust?",
    "What is Go used for?",
    "Which language is used for Apple application development?",
    "What is C# used for?",
    "What is Kotlin?"
]

selected_question = st.selectbox(
    "Choose an example question:",
    ["Select a question"] + example_questions
)


# ============================================================
# USER QUESTION
# ============================================================

st.subheader("🔍 Ask Your Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Python used for?"
)


# ============================================================
# USE EXAMPLE QUESTION
# ============================================================

if selected_question != "Select a question":

    if st.button("Use Selected Question"):

        question = selected_question

        st.session_state["question"] = question

        st.rerun()


if "question" in st.session_state:

    question = st.session_state["question"]


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🚀 Ask Question", type="primary"):

    if not question.strip():

        st.warning("⚠️ Please enter a question.")

    else:

        # ----------------------------------------------------
        # STEP 1 - SIMILARITY SEARCH
        # ----------------------------------------------------

        with st.spinner("🔎 Searching the knowledge base..."):

            search_results = vectorstore.similarity_search(
                question,
                k=3
            )


        # ----------------------------------------------------
        # STEP 2 - TOP-K RETRIEVAL
        # ----------------------------------------------------

        k = 2

        top_k_results = vectorstore.similarity_search(
            question,
            k=k
        )


        # ----------------------------------------------------
        # STEP 3 - CONTEXT CONSTRUCTION
        # ----------------------------------------------------

        context = "\n\n".join(
            result.page_content
            for result in top_k_results
        )


        # ----------------------------------------------------
        # STEP 4 - RAG PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are a college programming assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:

"I could not find this information in the college document."

Do not invent information.

Give a simple and clear answer suitable for a college student.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


        # ----------------------------------------------------
        # STEP 5 - GEMINI
        # ----------------------------------------------------

        with st.spinner("🤖 Generating answer..."):

            try:

                response = llm.invoke(prompt)

            except Exception as e:

                st.error("❌ Error while generating the answer.")

                st.exception(e)

                st.stop()


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        st.divider()

        st.subheader("💬 Answer")

        st.markdown(
            f"""
            <div class="answer-box">
            {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # RETRIEVED CONTEXT
        # ====================================================

        with st.expander("📄 View Retrieved Context"):

            for i, result in enumerate(top_k_results):

                st.markdown(
                    f"### Retrieved Chunk {i + 1}"
                )

                st.write(
                    result.page_content
                )

                st.divider()


        # ====================================================
        # SEARCH RESULTS
        # ====================================================

        with st.expander("🔎 View Similarity Search Results"):

            for i, result in enumerate(search_results):

                st.markdown(
                    f"### Search Result {i + 1}"
                )

                st.write(
                    result.page_content
                )

                st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 College Programming RAG | "
    "LangChain + HuggingFace + ChromaDB + Gemini"
)