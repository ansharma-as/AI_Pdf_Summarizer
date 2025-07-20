import streamlit as st
import tempfile
import os
from pathlib import Path

# Import our PDF QA system
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Choose your LLM
from langchain_community.llms import Ollama

@st.cache_resource
def initialize_components():
    """Initialize and cache LLM and embeddings"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    llm = Ollama(model="./mistral-7b-instruct-v0.1.Q4_K_M.gguf", temperature=0.7)
    
    return embeddings, llm

@st.cache_data
def process_pdf(_file_bytes, filename):
    """Process uploaded PDF and return chunks"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_file_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return chunks, len(documents)
        
    except Exception as e:
        os.unlink(tmp_path)
        raise e

def create_vectorstore(chunks, embeddings):
    """Create vector store from chunks"""
    return FAISS.from_documents(chunks, embeddings)

def answer_question(vectorstore, llm, question, k=5):
    """Generate answer for question"""
    # Retrieve relevant documents
    relevant_docs = vectorstore.similarity_search(question, k=k)
    
    if not relevant_docs:
        return "No relevant content found for your question.", []
    
    # Prepare context
    context_parts = []
    source_pages = set()
    
    for doc in relevant_docs:
        page_num = doc.metadata.get('page', 'Unknown')
        source_pages.add(page_num)
        context_parts.append(f"[Page {page_num}] {doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""Use the provided context to answer the question accurately.

CONTEXT:
{context}

QUESTION: {question}

Provide a detailed answer based only on the information in the context. Reference specific pages when possible.

ANSWER:"""

    # Get response
    response = llm.invoke(prompt)
    
    return response, sorted(list(source_pages))

def main():
    st.set_page_config(
        page_title="PDF Question Answering System",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Large PDF Question-Answering System")
    st.markdown("Upload a PDF (up to 200 pages) and ask questions about its content!")
    
    # Initialize components
    try:
        embeddings, llm = initialize_components()
        st.success("✅ AI components loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading AI components: {e}")
        st.stop()
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload PDF files up to 200MB"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 File: {uploaded_file.name}")
            st.info(f"📊 Size: {uploaded_file.size / (1024*1024):.1f} MB")
    
    # Main content area
    if uploaded_file is not None:
        # Process PDF
        if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            with st.spinner("🔄 Processing PDF... This may take a few minutes for large files."):
                try:
                    # Process the PDF
                    file_bytes = uploaded_file.read()
                    chunks, num_pages = process_pdf(file_bytes, uploaded_file.name)
                    
                    # Create vector store
                    vectorstore = create_vectorstore(chunks, embeddings)
                    
                    # Store in session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.num_pages = num_pages
                    st.session_state.num_chunks = len(chunks)
                    
                    st.success(f"✅ PDF processed successfully!")
                    st.info(f"📊 **{num_pages}** pages → **{len(chunks)}** searchable chunks")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
                    st.stop()
        
        # Display document info
        if "vectorstore" in st.session_state:
            with st.expander("📋 Document Information", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Pages", st.session_state.num_pages)
                with col2:
                    st.metric("📝 Chunks", st.session_state.num_chunks)
                with col3:
                    st.metric("📁 File", st.session_state.current_file)
        
        # Question answering interface
        st.header("💬 Ask Questions")
        
        # Question input
        question = st.text_input(
            "❓ What would you like to know about this document?",
            placeholder="e.g., What are the main conclusions? Summarize chapter 3..."
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options", expanded=False):
            num_sources = st.slider(
                "Number of sources to consider", 
                min_value=3, 
                max_value=10, 
                value=5,
                help="More sources = more comprehensive but slower"
            )
        
        # Answer generation
        if question and "vectorstore" in st.session_state:
            with st.spinner("🤔 Analyzing document and generating answer..."):
                try:
                    answer, source_pages = answer_question(
                        st.session_state.vectorstore, 
                        llm, 
                        question, 
                        k=num_sources
                    )
                    
                    # Display answer
                    st.subheader("🤖 Answer")
                    st.write(answer)
                    
                    # Display sources
                    if source_pages:
                        st.subheader("📚 Sources")
                        st.info(f"Based on content from pages: **{', '.join(map(str, source_pages))}**")
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
        
        # Sample questions
        if "vectorstore" in st.session_state:
            st.subheader("💡 Try these sample questions:")
            sample_questions = [
                "What are the main topics covered in this document?",
                "Can you provide a summary of the key findings?",
                "What are the conclusions or recommendations?",
                "Are there any important statistics or numbers mentioned?"
            ]
            
            col1, col2 = st.columns(2)
            for i, sample_q in enumerate(sample_questions):
                col = col1 if i % 2 == 0 else col2
                if col.button(sample_q, key=f"sample_{i}"):
                    st.rerun()
    
    else:
        # Welcome message
        st.info("👆 Please upload a PDF file using the sidebar to get started!")
        
        st.subheader("🌟 Features:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📄 Large PDF Support**
            - Handle PDFs up to 200 pages
            - Efficient processing and chunking
            - Smart text splitting
            """)
            
            st.markdown("""
            **🧠 Intelligent Q&A**
            - Context-aware answers
            - Page source references
            - Multiple retrieval sources
            """)
        
        with col2:
            st.markdown("""
            **⚡ Fast Processing**
            - Optimized for large documents
            - Cached embeddings
            - Vector similarity search
            """)
            
            st.markdown("""
            **🎯 Accurate Results**
            - AI-powered understanding
            - Contextual responses
            - Source attribution
            """)

if __name__ == "__main__":
    main()