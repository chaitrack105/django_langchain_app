from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
YOUR_GOOGLE_API_KEY = os.getenv('YOUR_GOOGLE_API_KEY')

class ContextAwareAssistant:
    def __init__(self, model_type="gemini"):
        # Initialize LLM
        if model_type == "openai":
            self.llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=YOUR_GOOGLE_API_KEY)
        
        # Initialize embeddings for context retrieval
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vectorstore = None
        
        # Memory for conversation history
        self.memory = ConversationBufferMemory(return_messages=True)
        
    def load_context_documents(self, file_paths):
        """Load and process documents for context"""
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file_path.endswith('.pdf'):
                loader = PyMuPDFLoader(file_path)
            else:
                continue
                
            docs = loader.load()
            documents.extend(docs)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
    
    def get_relevant_context(self, query, k=3):
        """Retrieve relevant context based on query"""
        if not self.vectorstore:
            return ""
        
        relevant_docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        return context
    
    def create_system_prompt(self, domain_context="", personality="helpful", expertise_areas=None):
        """Create customized system prompt for specific use cases"""
        
        base_prompt = f"You are a {personality} assistant"
        
        if domain_context:
            base_prompt += f" specialized in {domain_context}"
        
        if expertise_areas:
            expertise_list = ", ".join(expertise_areas)
            base_prompt += f" with expertise in: {expertise_list}"
        
        additional_instructions = """
        
Instructions:
- Provide accurate, helpful responses based on the context provided
- If you don't know something, say so clearly
- Use the conversation history to maintain context
- Be concise but thorough in your explanations
        """
        
        return base_prompt + additional_instructions

    def generate_response(self, user_input, history=None, use_context=True):
        """Generate response with context awareness"""
        
        # Get relevant context if available
        context = ""
        if use_context and self.vectorstore:
            context = self.get_relevant_context(user_input)
        
        # Create system prompt with context
        system_content = self.create_system_prompt()
        
        # Add context to system prompt if available
        if context:
            system_content += f"\n\nRelevant Context:\n{context}"
        
        # Build messages list starting with system message
        messages = [SystemMessage(content=system_content)]
        
        # Add conversation history
        if history:
            for msg in history:
                if hasattr(msg, 'sender'):  # Custom message object
                    if msg.sender == 'human':
                        messages.append(HumanMessage(content=msg.text))
                    elif msg.sender == 'ai':
                        messages.append(AIMessage(content=msg.text))
                elif isinstance(msg, (HumanMessage, AIMessage)):  # LangChain message objects
                    messages.append(msg)
                elif isinstance(msg, dict):  # Dictionary format
                    if msg.get('role') == 'user' or msg.get('type') == 'human':
                        messages.append(HumanMessage(content=msg.get('content', msg.get('text', ''))))
                    elif msg.get('role') == 'assistant' or msg.get('type') == 'ai':
                        messages.append(AIMessage(content=msg.get('content', msg.get('text', ''))))
        
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        # Generate response using the LLM directly
        try:
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Store in memory
            self.memory.save_context(
                {"input": user_input}, 
                {"output": response_text}
            )
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg

# Example usage for different contexts:

# 1. Customer Support Assistant
def create_support_assistant():
    assistant = ContextAwareAssistant()
    
    # Load FAQ documents, product manuals, etc.
    # assistant.load_context_documents(['faq.txt', 'product_manual.pdf'])
    
    assistant.create_system_prompt(
        domain_context="customer support",
        personality="friendly and solution-oriented",
        expertise_areas=["product troubleshooting", "billing", "technical support"]
    )
    
    return assistant

# 2. Medical Assistant (Educational purposes)
def create_medical_assistant():
    assistant = ContextAwareAssistant()
    
    # Load medical literature, guidelines
    # assistant.load_context_documents(['medical_guidelines.pdf', 'drug_database.txt'])
    
    assistant.create_system_prompt(
        domain_context="medical information (for educational purposes only)",
        personality="careful and precise",
        expertise_areas=["symptoms", "medications", "general health"]
    )
    
    return assistant

# 3. Code Assistant
def create_code_assistant():
    assistant = ContextAwareAssistant()
    
    # Load coding documentation, best practices
    # assistant.load_context_documents(['python_docs.txt', 'best_practices.md'])
    
    assistant.create_system_prompt(
        domain_context="software development",
        personality="technical and precise",
        expertise_areas=["Python", "JavaScript", "debugging", "architecture"]
    )
    
    return assistant

# Training/Fine-tuning approaches (Note: These are conceptual - actual training requires different setup)

class TrainingDataCollector:
    """Collect training data from conversations"""
    
    def __init__(self):
        self.training_data = []
    
    def collect_interaction(self, user_input, ai_response, context="", feedback_score=None):
        """Collect interaction data for training"""
        interaction = {
            "input": user_input,
            "output": ai_response,
            "context": context,
            "timestamp": time.time(),  # Fixed: was os.time.time()
            "feedback_score": feedback_score
        }
        self.training_data.append(interaction)
    
    def save_training_data(self, filename):
        """Save collected data for training"""
        with open(filename, 'w') as f:
            json.dump(self.training_data, f, indent=2)
    
    def export_for_fine_tuning(self, format_type="openai"):
        """Export in format suitable for fine-tuning"""
        if format_type == "openai":
            formatted_data = []
            for interaction in self.training_data:
                formatted_data.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": interaction["input"]},
                        {"role": "assistant", "content": interaction["output"]}
                    ]
                })
            return formatted_data

# Standalone function wrapper for easier usage
def generate_response(user_input, recent_messages=None, assistant_type="code", use_context=True):
    """
    Standalone function to generate responses - wrapper around ContextAwareAssistant
    
    Args:
        user_input (str): The user's input message
        recent_messages (list): List of recent conversation messages
        assistant_type (str): Type of assistant ("code", "support", "medical", or "general")
        use_context (bool): Whether to use context retrieval
    
    Returns:
        str: The AI assistant's response
    """
    # Create appropriate assistant based on type
    if assistant_type == "code":
        assistant = create_code_assistant()
    elif assistant_type == "support":
        assistant = create_support_assistant()
    elif assistant_type == "medical":
        assistant = create_medical_assistant()
    else:
        assistant = ContextAwareAssistant()
    
    # Generate and return response
    return assistant.generate_response(user_input, recent_messages, use_context)

# Example usage:
if __name__ == "__main__":
    # Method 1: Using the class directly
    assistant = create_code_assistant()
    response = assistant.generate_response("How do I handle exceptions in Python?")
    print("Class method response:", response)
    
    # Method 2: Using the standalone function
    response = generate_response("How do I handle exceptions in Python?", assistant_type="code")
    print("Function response:", response)
    
    # Method 3: With conversation history
    conversation_history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."}
    ]
    response = generate_response(
        "How do I handle exceptions in Python?", 
        recent_messages=conversation_history,
        assistant_type="code"
    )
    print("Response with history:", response)
    
    # For training data collection
    collector = TrainingDataCollector()
    collector.collect_interaction(
        "How do I handle exceptions in Python?",
        response,
        context="Programming tutorial",
        feedback_score=5
    )