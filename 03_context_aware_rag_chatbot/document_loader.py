import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma

class DocumentProcessor:
    def __init__(self, docs_path="documents", vectorstore_path="vectorstore"):
        self.docs_path = docs_path
        self.vectorstore_path = vectorstore_path
        self.embeddings = None
        self.vectorstore = None
    
    # Load all PDF files from documents directory
    def load_documents(self):
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    # Split documents into chunks for better retrieval
    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self):
        self.embeddings = CohereEmbeddings(
            model=os.getenv("COHERE_EMBEDDING_MODEL"),
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        return self.embeddings
    
    # Create or load vector store using Chroma
    def create_vectorstore(self, chunks):
        if not self.embeddings:
            self.embeddings = self.create_embeddings()
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vectorstore_path
        )
        self.vectorstore.persist()
        print(f"Vector store created at {self.vectorstore_path}")
        return self.vectorstore
    
    # Load existing vector store
    def load_vectorstore(self):
        if not self.embeddings:
            self.embeddings = self.create_embeddings()
        
        if os.path.exists(self.vectorstore_path):
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings
            )
            print("Vector store loaded successfully")
            return self.vectorstore
        else:
            print("Vector store not found. Please create it first.")
            return None
    
    def process_documents(self):
        # pipeline: load -> split -> create vectorstore
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        vectorstore = self.create_vectorstore(chunks)
        return vectorstore
