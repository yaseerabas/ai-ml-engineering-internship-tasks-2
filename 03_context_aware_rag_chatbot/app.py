import streamlit as st
import os
from document_loader import DocumentProcessor
from chatbot import RAGChatbot


st.set_page_config(
    page_title="Context Aware RAG Chatbot",
    layout="wide"
)

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False

# Load existing vectorstore
def load_vectorstore():
    processor = DocumentProcessor()
    vectorstore = processor.load_vectorstore()
    return vectorstore

# Process documents and create vectorstore
def process_documents():
    processor = DocumentProcessor()
    with st.spinner("Processing documents... This may take a few minutes."):
        vectorstore = processor.process_documents()
    return vectorstore

# Sidebar for document management
with st.sidebar:
    st.title("Document Management")
    
    st.markdown("### Setup Instructions:")
    st.markdown("""
    1. Add PDF files to `documents/` folder
    2. Click 'Process Documents' button
    3. Start chatting!
    """)
    
    # Check if documents folder exists
    if not os.path.exists("documents"):
        os.makedirs("documents")
        st.warning("Created 'documents' folder. Please add your PDF files there.")
    
    # Document processing button
    if st.button("Process Documents", use_container_width=True):
        if os.path.exists("documents") and len(os.listdir("documents")) > 0:
            st.session_state.vectorstore = process_documents()
            st.session_state.vectorstore_loaded = True
            st.success("Documents processed successfully!")
            st.rerun()
        else:
            st.error("Please add PDF files to the documents folder first!")
    
    # Load existing vectorstore button
    if st.button("📂 Load Existing Vectorstore", use_container_width=True):
        vectorstore = load_vectorstore()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.vectorstore_loaded = True
            st.success("Vectorstore loaded successfully!")
            st.rerun()
        else:
            st.error("No vectorstore found. Please process documents first.")
    
    st.markdown("---")
    
    # Clear chat history button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_history()
        st.success("Chat history cleared!")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Status")
    if st.session_state.vectorstore_loaded:
        st.success("Vectorstore loaded")
    else:
        st.info("Vectorstore not loaded")

# Main chat interface
st.title("Context-Aware RAG Chatbot")
st.markdown("Ask questions about your documents and get intelligent answers with context memory!")

# Check if vectorstore is loaded
if not st.session_state.vectorstore_loaded:
    st.info("👈 Please load or process documents from the sidebar to start chatting.")
    st.stop()

# Initialize chatbot if not already done
if st.session_state.chatbot is None and st.session_state.vectorstore:
    st.session_state.chatbot = RAGChatbot(vectorstore=st.session_state.vectorstore)
    st.session_state.chatbot.setup_conversation_chain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander(f"📄 View Sources ({len(message['sources'])} documents)"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.chat(prompt)
            answer = response["answer"]
            sources = response["sources"]
        
        st.markdown(answer)
        
        if sources:
            with st.expander(f"📄 View Sources ({len(sources)} documents)"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(doc.page_content[:300] + "...")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.caption(f"Metadata: {doc.metadata}")
                    st.markdown("---")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with LangChain, Streamlit, and ChromaDB | Context-aware RAG system"
    "</div>",
    unsafe_allow_html=True
)
