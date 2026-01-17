import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from document_loader import DocumentProcessor

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore
        self.memory = None
        self.conversation_chain = None
        self.chat_history = []
        
    def initialize_llm(self):
        # Initialize OpenAI chat model
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        llm = ChatOpenAI(
            temperature=0.7,
            model_name=model_name,
            openai_api_key=api_key,
        )
        return llm
    
    def initialize_memory(self):
        # Create conversation memory to remember context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        return self.memory
    
    def create_prompt_template(self):
        # Custom prompt for the chatbot
        template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always try to be helpful and provide detailed answers based on the context provided.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        return prompt
    
    def setup_conversation_chain(self):
        # Setup the conversational retrieval chain
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        llm = self.initialize_llm()
        self.memory = self.initialize_memory()
        
        # Create retriever from vectorstore
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
        )
        
        # Create conversational chain with memory
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
        
        return self.conversation_chain
    
    def chat(self, question):
        # Send a question and get response with context
        if not self.conversation_chain:
            self.setup_conversation_chain()
        
        response = self.conversation_chain({"question": question})
        
        # Extract answer and source documents
        answer = response["answer"]
        source_docs = response.get("source_documents", [])
        
        return {
            "answer": answer,
            "sources": source_docs
        }
    
    def get_chat_history(self):
        # Return the conversation history
        if self.memory:
            return self.memory.load_memory_variables({})
        return []
    
    def clear_history(self):
        # Clear conversation memory
        if self.memory:
            self.memory.clear()
        self.chat_history = []
        print("Chat history cleared")