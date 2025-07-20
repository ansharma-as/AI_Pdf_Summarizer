# from langchain.llms import LlamaCpp
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import load_qa_chain
# from langchain.prompts import PromptTemplate
 
# import os
 
# # === Model config ===
# MODEL_PATH = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# PDF_PATH = "./document.pdf"
 
# # === Load LLM ===
# llm = LlamaCpp(
#     model_path=MODEL_PATH,
#     temperature=0.7,
#     max_tokens=512,
#     top_p=1,
#     n_ctx=2048,
#     n_batch=512,
#     f16_kv=True,
#     verbose=True,
# )
 
# # === Load and split PDF ===
# loader = PyPDFLoader(PDF_PATH)
# documents = loader.load()
 
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)
 
# # === Embedding & Vectorstore ===
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(docs, embedding)
 
# # === Prompt Template ===
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a helpful assistant. Use the following context to answer the question.
# If you don't know the answer, say you don't know.
 
# Context:
# {context}
 
# Question: {question}
# Answer:
# """,
# )
 
# # === Load QA chain ===
# chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
 
# # === Interactive Q&A loop ===
# print("🤖 Ask questions about the PDF. Type 'exit' to quit.\n")
 
# while True:
#     query = input("❓ Your question: ")
#     if query.lower() == "exit":
#         break
 
#     # Fetch relevant docs
#     relevant_docs = vectorstore.similarity_search(query, k=3)
 
#     # Debug info
#     print("\n📄 Top relevant context chunk:\n", relevant_docs[0].page_content[:300], "\n")
 
#     # Run the chain
#     try:
#         response = chain.invoke({
#             "input_documents": relevant_docs,
#             "question": query
#         })
#         print("🤖 Answer:", response['output_text'])
#     except Exception as e:
#         print("❌ Error during inference:", str(e))
 




# from langchain_community.llms import LlamaCpp
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# import os

# MODEL_PATH = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# PDF_PATH = "./document.pdf"

# llm = LlamaCpp(
#     model_path=MODEL_PATH,
#     temperature=0.7,
#     max_tokens=512,
#     top_p=1,
#     n_ctx=2048,
#     n_batch=512,
#     f16_kv=True,
#     verbose=True,
# )

# loader = PyPDFLoader(PDF_PATH)
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)

# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(docs, embedding)

# print("🤖 Ask questions about the PDF. Type 'exit' to quit.\n")

# while True:
#     query = input("❓ Your question: ")
#     if query.lower() == "exit":
#         break

#     try:
#         relevant_docs = vectorstore.similarity_search(query, k=3)
        
#         print(f"\n📄 Top relevant context chunk:\n{relevant_docs[0].page_content[:300]}\n")
        
#         context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
#         prompt = f"""
# You are a helpful assistant. Use the following context to answer the question.
# If you don't know the answer, say you don't know.

# Context:
# {context}

# Question: {query}
# Answer:
# """
        
#         response = llm.invoke(prompt)
#         print("🤖 Answer:", response)
        
#     except Exception as e:
#         print("❌ Error during inference:", str(e))



from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import re
from typing import List, Dict

MODEL_PATH = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"
PDF_PATH = "./document.pdf"

# Enhanced LLM configuration for better performance
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,  # Lower temperature for more focused answers
    max_tokens=1024,  # Increased for more comprehensive answers
    top_p=0.95,
    n_ctx=4096,  # Increased context window
    n_batch=512,
    f16_kv=True,
    verbose=False,  # Set to False for cleaner output
    repeat_penalty=1.1,  # Prevent repetition
)

print("🔄 Loading and processing PDF...")

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Enhanced text splitting for better context preservation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Larger chunks for better context
    chunk_overlap=100,  # More overlap for continuity
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting points
)

docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)

print(f"✅ PDF processed successfully!")
print(f"📊 Loaded {len(documents)} pages into {len(docs)} searchable chunks")
print("🤖 Ask questions about the PDF. Type 'exit' to quit.\n")

def clean_response(response: str) -> str:
    """Clean and format the LLM response"""
    # Remove any repeated phrases
    lines = response.split('\n')
    cleaned_lines = []
    seen_lines = set()
    
    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            cleaned_lines.append(line)
            seen_lines.add(line)
    
    return '\n'.join(cleaned_lines).strip()

def get_enhanced_context(relevant_docs: List, query: str) -> tuple:
    """Create enhanced context with page references and relevance scoring"""
    context_parts = []
    page_refs = set()
    
    for i, doc in enumerate(relevant_docs):
        page_num = doc.metadata.get('page', 'Unknown')
        page_refs.add(page_num)
        
        # Add context with clear demarcation
        context_part = f"[Source {i+1} - Page {page_num}]\n{doc.page_content.strip()}"
        context_parts.append(context_part)
    
    full_context = "\n\n---\n\n".join(context_parts)
    return full_context, sorted(list(page_refs))

def create_advanced_prompt(context: str, query: str, page_refs: List) -> str:
    """Create an advanced prompt template for better responses"""
    
    prompt = f"""<|im_start|>system
You are an expert document analyst with advanced reading comprehension skills. Your task is to provide accurate, comprehensive, and well-structured answers based strictly on the provided document context.

GUIDELINES:
- Analyze the context thoroughly before responding
- Provide detailed, informative answers when the information is available
- Be specific and cite relevant details from the context
- If information is incomplete, clearly state what is missing
- Structure your response logically with clear explanations
- Use professional, clear language
- Reference specific sections or pages when relevant
- If the context doesn't contain the answer, honestly state this limitation

CONTEXT FROM DOCUMENT:
{context}

AVAILABLE SOURCES: Pages {', '.join(map(str, page_refs))}
<|im_end|>

<|im_start|>user
Based on the document context provided above, please answer the following question comprehensively:

{query}

Provide a detailed response that:
1. Directly addresses the question
2. Includes relevant details and examples from the context
3. Mentions specific page references where applicable
4. Indicates if any important information might be missing
<|im_end|>

<|im_start|>assistant
Based on my analysis of the document content, here is my comprehensive response:

"""
    
    return prompt

def analyze_query_type(query: str) -> str:
    """Analyze the type of query to adjust retrieval strategy"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['summary', 'summarize', 'overview', 'main points']):
        return 'summary'
    elif any(word in query_lower for word in ['definition', 'what is', 'explain', 'define']):
        return 'definition'
    elif any(word in query_lower for word in ['how', 'process', 'steps', 'procedure']):
        return 'process'
    elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because']):
        return 'reasoning'
    elif any(word in query_lower for word in ['when', 'date', 'time', 'year']):
        return 'temporal'
    elif any(word in query_lower for word in ['where', 'location', 'place']):
        return 'location'
    elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
        return 'comparison'
    elif any(word in query_lower for word in ['list', 'enumerate', 'bullet', 'points']):
        return 'list'
    else:
        return 'general'

def get_adaptive_k(query_type: str) -> int:
    """Get adaptive number of documents based on query type"""
    k_mapping = {
        'summary': 6,     # Need more context for summaries
        'comparison': 5,  # Need multiple sources for comparisons
        'list': 4,        # Need good coverage for lists
        'process': 4,     # Need sequential information
        'general': 3,     # Standard retrieval
        'definition': 2,  # Usually need focused information
        'temporal': 3,    # Time-specific information
        'location': 3,    # Location-specific information
        'reasoning': 4,   # Need context for explanations
    }
    return k_mapping.get(query_type, 3)

# Main interaction loop
while True:
    query = input("❓ Your question: ")
    if query.lower() == "exit":
        break

    try:
        # Analyze query type for adaptive retrieval
        query_type = analyze_query_type(query)
        k_value = get_adaptive_k(query_type)
        
        print(f"🔍 Analyzing query (Type: {query_type}, Sources: {k_value})...")
        
        # Retrieve relevant documents
        relevant_docs = vectorstore.similarity_search(query, k=k_value)
        
        if not relevant_docs:
            print("❌ No relevant content found for your question.")
            continue
        
        # Get enhanced context and page references
        context, page_refs = get_enhanced_context(relevant_docs, query)
        
        # Show preview of sources
        print(f"📄 Found relevant content from pages: {', '.join(map(str, page_refs))}")
        print(f"📝 Context preview: {relevant_docs[0].page_content[:200]}...\n")
        
        # Create advanced prompt
        prompt = create_advanced_prompt(context, query, page_refs)
        
        # Generate response
        print("🤔 Generating comprehensive answer...")
        response = llm.invoke(prompt)
        
        # Clean and format response
        cleaned_response = clean_response(response)
        
        # Display results
        print("🤖 Answer:")
        print("=" * 60)
        print(cleaned_response)
        print("=" * 60)
        print(f"📚 Sources: Pages {', '.join(map(str, page_refs))} | Query Type: {query_type}")
        print(f"📊 Based on {len(relevant_docs)} document sections\n")
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        print("💡 Try rephrasing your question or check if the PDF was loaded correctly.\n")



# import os
# import time
# from typing import List, Dict, Any
# from pathlib import Path

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document

# # Choose your LLM (uncomment one):
# # Option 1: Ollama (recommended for ease of use)
# # from langchain_community.llms import Ollama
# # def get_llm():
# #     return Ollama(model="./mistral-7b-instruct-v0.1.Q4_K_M.gguf", temperature=0.7)

# # Option 2: LlamaCpp (if you have llama-cpp-python installed)
# from langchain_community.llms import LlamaCpp
# def get_llm():
#     return LlamaCpp(
#         model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#         temperature=0.7,
#         max_tokens=512,
#         top_p=1,
#         n_ctx=4096,  # Increased context for large docs
#         n_batch=512,
#         f16_kv=True,
#         verbose=False,
#     )

# class LargePDFQASystem:
#     def __init__(self):
#         self.vectorstore = None
#         self.llm = None
#         self.embeddings = None
#         self.document_metadata = {}
        
#     def initialize_components(self):
#         """Initialize LLM and embeddings"""
#         print("🔧 Initializing AI components...")
        
#         # Initialize embeddings
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': False}
#         )
        
#         # Initialize LLM
#         try:
#             self.llm = get_llm()
#             print("✅ AI components initialized successfully!")
#         except Exception as e:
#             print(f"❌ Error initializing LLM: {e}")
#             print("💡 Make sure you have Ollama installed and running, or llama-cpp-python with model files")
#             return False
#         return True
    
#     def process_large_pdf(self, pdf_path: str) -> bool:
#         """Process a large PDF file efficiently"""
#         if not os.path.exists(pdf_path):
#             print(f"❌ PDF file not found: {pdf_path}")
#             return False
            
#         print(f"📄 Processing PDF: {pdf_path}")
#         start_time = time.time()
        
#         try:
#             # Load PDF
#             print("🔄 Loading PDF...")
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
            
#             print(f"📊 Loaded {len(documents)} pages")
            
#             # Store document metadata
#             self.document_metadata = {
#                 'filename': os.path.basename(pdf_path),
#                 'total_pages': len(documents),
#                 'total_chars': sum(len(doc.page_content) for doc in documents)
#             }
            
#             # Optimize text splitting for large documents
#             print("✂️ Splitting text into chunks...")
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=800,  # Larger chunks for better context
#                 chunk_overlap=100,  # Good overlap for continuity
#                 length_function=len,
#                 separators=["\n\n", "\n", ". ", " ", ""]
#             )
            
#             # Split documents
#             chunks = text_splitter.split_documents(documents)
#             print(f"📝 Created {len(chunks)} text chunks")
            
#             # Create vector store
#             print("🧠 Creating vector embeddings...")
#             self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
#             # Save vectorstore for future use
#             vectorstore_path = f"vectorstore_{os.path.splitext(os.path.basename(pdf_path))[0]}"
#             self.vectorstore.save_local(vectorstore_path)
#             print(f"💾 Vector store saved to: {vectorstore_path}")
            
#             end_time = time.time()
#             print(f"✅ PDF processed successfully in {end_time - start_time:.2f} seconds")
#             return True
            
#         except Exception as e:
#             print(f"❌ Error processing PDF: {e}")
#             return False
    
#     def load_existing_vectorstore(self, vectorstore_path: str) -> bool:
#         """Load a previously created vector store"""
#         try:
#             if os.path.exists(vectorstore_path):
#                 print(f"📂 Loading existing vector store from: {vectorstore_path}")
#                 self.vectorstore = FAISS.load_local(
#                     vectorstore_path, 
#                     self.embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#                 print("✅ Vector store loaded successfully!")
#                 return True
#             else:
#                 print(f"❌ Vector store not found: {vectorstore_path}")
#                 return False
#         except Exception as e:
#             print(f"❌ Error loading vector store: {e}")
#             return False
    
#     def answer_question(self, question: str, k: int = 5) -> Dict[str, Any]:
#         """Answer a question based on the PDF content"""
#         if not self.vectorstore or not self.llm:
#             return {"error": "System not properly initialized"}
        
#         try:
#             # Retrieve relevant documents
#             relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
#             if not relevant_docs:
#                 return {"error": "No relevant content found for your question"}
            
#             # Prepare context with page information
#             context_parts = []
#             source_pages = set()
            
#             for doc in relevant_docs:
#                 page_num = doc.metadata.get('page', 'Unknown')
#                 source_pages.add(page_num)
#                 context_parts.append(f"[Page {page_num}] {doc.page_content}")
            
#             context = "\n\n".join(context_parts)
            
#             # Create an enhanced prompt
#             prompt = f"""You are an AI assistant analyzing a document. Use the provided context to answer the question accurately and comprehensively.

# CONTEXT FROM DOCUMENT:
# {context}

# QUESTION: {question}

# INSTRUCTIONS:
# - Provide a detailed, accurate answer based only on the information in the context
# - If the answer isn't fully contained in the context, say so clearly
# - Reference specific pages when possible
# - Be comprehensive but concise
# - If you're uncertain about any part of the answer, mention it

# ANSWER:"""

#             # Get response from LLM
#             response = self.llm.invoke(prompt)
            
#             return {
#                 "answer": response,
#                 "source_pages": sorted(list(source_pages)),
#                 "num_sources": len(relevant_docs),
#                 "context_preview": context[:300] + "..." if len(context) > 300 else context
#             }
            
#         except Exception as e:
#             return {"error": f"Error generating answer: {e}"}
    
#     def get_document_summary(self) -> str:
#         """Get a summary of the loaded document"""
#         if not self.document_metadata:
#             return "No document loaded"
        
#         return f"""📄 Document Information:
# • Filename: {self.document_metadata['filename']}
# • Total Pages: {self.document_metadata['total_pages']}
# • Total Characters: {self.document_metadata['total_chars']:,}
# • Status: Ready for questions"""

# def main():
#     """Main application loop"""
#     print("🚀 Large PDF Question-Answering System")
#     print("=" * 50)
    
#     qa_system = LargePDFQASystem()
    
#     # Initialize components
#     if not qa_system.initialize_components():
#         return
    
#     while True:
#         print("\n📋 Options:")
#         print("1. Upload and process a new PDF")
#         print("2. Load existing processed PDF")
#         print("3. Ask questions about loaded PDF")
#         print("4. Show document info")
#         print("5. Exit")
        
#         choice = input("\n👆 Select an option (1-5): ").strip()
        
#         if choice == "1":
#             pdf_path = input("📁 Enter PDF file path: ").strip().replace('"', '')
#             if qa_system.process_large_pdf(pdf_path):
#                 print(f"\n{qa_system.get_document_summary()}")
        
#         elif choice == "2":
#             vectorstore_path = input("📂 Enter vector store path: ").strip()
#             qa_system.load_existing_vectorstore(vectorstore_path)
        
#         elif choice == "3":
#             if not qa_system.vectorstore:
#                 print("❌ No PDF loaded. Please upload a PDF first.")
#                 continue
            
#             print("\n💬 Ask questions about your PDF (type 'back' to return to menu)")
#             while True:
#                 question = input("\n❓ Your question: ").strip()
#                 if question.lower() == 'back':
#                     break
#                 if not question:
#                     continue
                
#                 print("🤔 Thinking...")
#                 result = qa_system.answer_question(question)
                
#                 if "error" in result:
#                     print(f"❌ {result['error']}")
#                 else:
#                     print(f"\n🤖 Answer:")
#                     print(result['answer'])
#                     print(f"\n📚 Sources: Pages {', '.join(map(str, result['source_pages']))}")
#                     print(f"📊 Based on {result['num_sources']} relevant sections")
        
#         elif choice == "4":
#             print(f"\n{qa_system.get_document_summary()}")
        
#         elif choice == "5":
#             print("👋 Goodbye!")
#             break
        
#         else:
#             print("❌ Invalid option. Please try again.")

# if __name__ == "__main__":
#     main()