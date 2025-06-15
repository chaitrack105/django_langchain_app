# utils.py - Enhanced with MySQL Integration (CORRECTED)

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
import os
import sys
from dotenv import load_dotenv
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MySQL integration classes
from simplified_mysql_integration import (
    SimplifiedMySQLLeadManager, 
    SimplifiedMySQLLeadScoringTrainer,
    DatabaseConfig,
    create_mysql_lead_scoring_assistant,
    quick_train_and_predict
)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
YOUR_GOOGLE_API_KEY = os.getenv('YOUR_GOOGLE_API_KEY')

# Database configuration from environment
DB_CONFIG = DatabaseConfig.from_env()

class ContextAwareAssistant:
    def __init__(self, model_type="gemini", use_mysql=False, db_config=None):
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
        
        # MySQL integration
        self.use_mysql = use_mysql
        self.mysql_trainer = None
        self.model_trained = False  # Track if model is trained
        
        if use_mysql:
            try:
                self.db_config = db_config or DB_CONFIG
                self.mysql_trainer = SimplifiedMySQLLeadScoringTrainer(self.db_config)
                print(f"✅ MySQL integration enabled for database: {self.db_config.database}")
            except Exception as e:
                print(f"❌ MySQL integration failed: {e}")
                self.use_mysql = False
                self.mysql_trainer = None
    
    def ensure_model_trained(self):
        """Ensure model is trained before use"""
        if not self.model_trained and self.use_mysql and self.mysql_trainer:
            try:
                print("🔄 Auto-training model...")
                results = self.mysql_trainer.train_from_database(limit=100)  # Use limited data for faster training
                self.model_trained = True
                print("✅ Model auto-trained successfully")
                return True
            except Exception as e:
                print(f"❌ Auto-training failed: {e}")
                return False
        return self.model_trained
            
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
        
        # Add lead scoring context if this is a lead prioritization assistant
        if "lead" in domain_context.lower():
            base_prompt += """
            
You have access to a comprehensive lead scoring system with the following components:

BEAT PLAN TAGS:
- ERV_Active: Early Revenue Verification - Active accounts
- ERV_Winback: Early Revenue Verification - Win Back campaigns  
- ERV_Dormant: Early Revenue Verification - Dormant accounts
- NCA_Hot: New Customer Acquisition - Hot leads
- NCA_Warm: New Customer Acquisition - Warm leads
- NCA_Cold: New Customer Acquisition - Cold leads

TURNOVER RANGES:
- Rs. 10cr - Rs.50cr: Small enterprises
- Rs. 50cr - Rs.100cr: Medium enterprises
- Rs. 100cr - Rs.500cr: Large enterprises
- Rs. 500cr - Rs.1000cr: Enterprise accounts

PRODUCTS: Iron rod, HR Sheet, TMT Bar, Wire Rod, Steel Coil, Cement

You can score leads, provide rankings, and suggest prioritization strategies using real database data.
            """
        
        additional_instructions = """
        
Instructions:
- Provide accurate, helpful responses based on the context provided
- If you don't know something, say so clearly
- Use the conversation history to maintain context
- Be concise but thorough in your explanations
- For lead scoring queries, use the MySQL database when available
- Always prioritize database results over general knowledge for lead-specific queries
        """
        
        return base_prompt + additional_instructions

    def train_model_from_database(self, force_retrain=False):
        """Train the scoring model using database data"""
        if not self.use_mysql or not self.mysql_trainer:
            return {"error": "MySQL integration not enabled"}
        
        if self.model_trained and not force_retrain:
            return {"status": "success", "message": "Model already trained"}
        
        try:
            results = self.mysql_trainer.train_from_database()
            self.model_trained = True
            return {
                "status": "success",
                "message": "Model trained successfully",
                "performance": results
            }
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}

    def score_lead_from_database(self, prospect_id: str):
        """Score a lead using the trained model and database data"""
        if not self.use_mysql or not self.mysql_trainer:
            return {"error": "MySQL integration not enabled"}
        
        # Ensure model is trained
        if not self.ensure_model_trained():
            return {"error": "Model training failed"}
        
        try:
            result = self.mysql_trainer.predict_for_prospect(prospect_id)
            return {
                "status": "success",
                "prediction": result
            }
        except Exception as e:
            return {"error": f"Scoring failed: {str(e)}"}

    def get_top_leads(self, limit: int = 10):
        """Get top scored leads from database"""
        if not self.use_mysql or not self.mysql_trainer:
            return {"error": "MySQL integration not enabled"}
        
        # Ensure model is trained
        if not self.ensure_model_trained():
            return {"error": "Model training failed"}
        
        try:
            all_predictions = self.mysql_trainer.batch_predict_all_leads()
            top_leads = all_predictions.head(limit)
            return {
                "status": "success",
                "top_leads": top_leads.to_dict('records')
            }
        except Exception as e:
            return {"error": f"Failed to get top leads: {str(e)}"}

    def get_database_leads_by_criteria(self, criteria: dict):
        """Get leads from database based on criteria"""
        if not self.use_mysql or not self.mysql_trainer:
            return {"error": "MySQL integration not enabled"}
        
        try:
            # Build query based on criteria
            where_conditions = []
            params = {}
            
            if 'beat_tag' in criteria:
                where_conditions.append("beat_tag = %(beat_tag)s")
                params['beat_tag'] = criteria['beat_tag']
            
            if 'turnover' in criteria:
                where_conditions.append("turnover = %(turnover)s")
                params['turnover'] = criteria['turnover']
            
            if 'min_intent_score' in criteria:
                where_conditions.append("lead_intent_score >= %(min_intent_score)s")
                params['min_intent_score'] = criteria['min_intent_score']
            
            if 'city' in criteria:
                where_conditions.append("address LIKE %(city)s")
                params['city'] = f"%{criteria['city']}%"
            
            # Base query
            query = "SELECT * FROM nx_op_ld_ai_lead360"
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            query += " LIMIT 20"  # Limit results
            
            # Debug: Print the query and parameters
            print(f"🔍 Debug Query: {query}")
            print(f"🔍 Debug Params: {params}")
            print(f"🔍 Debug Criteria: {criteria}")
            
            import pandas as pd
            results_df = pd.read_sql(query, self.mysql_trainer.db_manager.engine, params=params)
            
            return {
                "status": "success",
                "leads": results_df.to_dict('records'),
                "count": len(results_df),
                "query": query,
                "params": params
            }
            
        except Exception as e:
            return {"error": f"Database query failed: {str(e)}"}

    def debug_beat_tags(self):
        """Debug function to see all unique beat tags in database"""
        if not self.use_mysql or not self.mysql_trainer:
            return "MySQL not enabled"
        
        try:
            import pandas as pd
            query = "SELECT DISTINCT beat_tag FROM nx_op_ld_ai_lead360"
            results = pd.read_sql(query, self.mysql_trainer.db_manager.engine)
            return list(results['beat_tag'])
        except Exception as e:
            return f"Error: {e}"
    def generate_response(self, user_input, history=None, use_context=True):
        """Generate response with context awareness and MySQL lead scoring"""
        
        # PRIORITY 1: Handle MySQL-specific commands first
        if self.use_mysql and any(keyword in user_input.lower() for keyword in 
                                 ['train model', 'score lead', 'prospect', 'top leads', 'database']):
            return self._handle_mysql_commands(user_input)
        
        # PRIORITY 2: Handle lead-related queries with database integration
        if self.use_mysql and any(keyword in user_input.lower() for keyword in 
                                ['lead', 'priority', 'rank', 'beat', 'turnover', 'intent']):
            
            # Try to extract prospect ID first
            prospect_id = self._extract_prospect_id(user_input)
            if prospect_id:
                result = self.score_lead_from_database(prospect_id)
                if result.get("status") == "success":
                    return self._format_lead_scoring_response(result["prediction"])
                else:
                    return f"❌ Could not score lead {prospect_id}: {result.get('error')}"
            
            # Try to extract filtering criteria
            criteria = self._extract_lead_criteria(user_input)
            if criteria:
                result = self.get_database_leads_by_criteria(criteria)
                if result.get("status") == "success":
                    return self._format_leads_list_response(result["leads"], criteria)
            
            # For general lead queries, get top leads
            if any(term in user_input.lower() for term in ['best', 'top', 'highest', 'priority']):
                result = self.get_top_leads(10)
                if result.get("status") == "success":
                    return self._format_top_leads_response(result["top_leads"])

        # PRIORITY 3: Regular LLM response for non-lead queries
        return self._generate_regular_response(user_input, history, use_context)

    def _generate_regular_response(self, user_input, history, use_context):
        """Generate regular LLM response"""
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

    def _extract_lead_criteria(self, user_input: str) -> dict:
        """Extract lead filtering criteria from user input"""
        criteria = {}
        user_input_lower = user_input.lower()
        
        # Extract beat tag - match exact database values
        beat_tag_mappings = {
            'erv win back': 'ERV win back',
            'win back': 'ERV win back', 
            'winback': 'ERV win back',
            'active': 'Active',
            'new lead': 'New Lead',
            'new': 'New Lead',
            'dormant': 'Dormant'
        }
        
        for search_term, db_value in beat_tag_mappings.items():
            if search_term in user_input_lower:
                criteria['beat_tag'] = db_value
                break
        
        # Extract turnover - match exact database values
        turnover_mappings = {
            '10cr - 50cr': 'Rs. 10cr - Rs.50cr',
            '10-50': 'Rs. 10cr - Rs.50cr',
            '50cr - 100cr': 'Rs. 50cr - Rs.100cr', 
            '50-100': 'Rs. 50cr - Rs.100cr',
            '100cr - 500cr': 'Rs. 100cr - Rs.500cr',
            '100-500': 'Rs. 100cr - Rs.500cr',
            '500cr - 1000cr': 'Rs. 500cr - Rs.1000cr',
            '500-1000': 'Rs. 500cr - Rs.1000cr'
        }
        
        for search_term, db_value in turnover_mappings.items():
            if search_term in user_input_lower:
                criteria['turnover'] = db_value
                break
        
        # Extract intent score
        import re
        intent_match = re.search(r'intent.*?(\d+)', user_input_lower)
        if intent_match:
            criteria['min_intent_score'] = int(intent_match.group(1))
        
        # Extract city from common Indian cities
        cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'pune', 'hyderabad', 
                 'gurgaon', 'noida', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'patna']
        for city in cities:
            if city in user_input_lower:
                criteria['city'] = city.title()
                break
        
        return criteria

    def _format_leads_list_response(self, leads: list, criteria: dict) -> str:
        """Format leads list response"""
        if not leads:
            return f"❌ No leads found matching criteria: {criteria}"
        
        # Show what criteria were applied
        criteria_text = []
        for key, value in criteria.items():
            criteria_text.append(f"{key}: {value}")
        
        response = f"📋 **Found {len(leads)} leads"
        if criteria_text:
            response += f" matching criteria ({', '.join(criteria_text)})"
        response += ":**\n\n"
        
        for i, lead in enumerate(leads[:10], 1):  # Show max 10
            response += f"{i}. **{lead['prospectId']}**\n"
            response += f"   Beat Tag: {lead['beat_tag']}\n"
            response += f"   Turnover: {lead['turnover']}\n"
            response += f"   Intent Score: {lead['lead_intent_score']}/100\n"
            response += f"   Profile Score: {lead['lead_profile_score']}/100\n"
            response += f"   Location: {lead['address'][:50]}...\n\n"
        
        if len(leads) > 10:
            response += f"... and {len(leads) - 10} more leads\n"
        
        return response

    def _format_top_leads_response(self, leads: list) -> str:
        """Format top leads response"""
        response = f"🏆 **Top {len(leads)} Leads by Predicted Score:**\n\n"
        for i, lead in enumerate(leads, 1):
            response += f"{i}. **{lead['prospect_id']}** - Score: {lead['predicted_score']:.2f}\n"
            response += f"   Beat Tag: {lead['input_data']['beat_tag']}\n"
            response += f"   Turnover: {lead['input_data']['turnover']}\n\n"
        return response

    def _handle_mysql_commands(self, user_input: str) -> str:
        """Handle MySQL-specific commands"""
        user_input_lower = user_input.lower()
        
        if "train model" in user_input_lower:
            result = self.train_model_from_database()
            if result.get("status") == "success":
                performance = result["performance"]
                response = "✅ **Model Training Completed**\n\n"
                for model_name, metrics in performance.items():
                    response += f"**{model_name.upper()}:**\n"
                    response += f"- R² Score: {metrics['r2']:.3f}\n"
                    response += f"- Mean Absolute Error: {metrics['mae']:.3f}\n\n"
                return response
            else:
                return f"❌ Training failed: {result.get('error')}"
        
        elif "top leads" in user_input_lower:
            # Extract number if specified
            import re
            numbers = re.findall(r'\d+', user_input)
            limit = int(numbers[0]) if numbers else 10
            
            result = self.get_top_leads(limit)
            if result.get("status") == "success":
                leads = result["top_leads"]
                response = f"🏆 **Top {len(leads)} Leads:**\n\n"
                for i, lead in enumerate(leads, 1):
                    response += f"{i}. **{lead['prospect_id']}** - Score: {lead['predicted_score']:.2f}\n"
                    response += f"   Beat Tag: {lead['input_data']['beat_tag']}\n"
                    response += f"   Turnover: {lead['input_data']['turnover']}\n\n"
                return response
            else:
                return f"❌ Failed to get top leads: {result.get('error')}"
        
        elif "score lead" in user_input_lower or "prospect" in user_input_lower:
            prospect_id = self._extract_prospect_id(user_input)
            if prospect_id:
                result = self.score_lead_from_database(prospect_id)
                if result.get("status") == "success":
                    return self._format_lead_scoring_response(result["prediction"])
                else:
                    return f"❌ Scoring failed: {result.get('error')}"
            else:
                return "Please specify a prospect ID (e.g., 'score lead abc123')"
        
        return "Available MySQL commands: 'train model', 'top leads', 'score lead [prospect_id]'"

    def _extract_prospect_id(self, user_input: str) -> str:
        """Extract prospect ID from user input"""
        import re
        # Look for patterns like abc123, LEAD_001, etc.
        pattern = r'\b([a-zA-Z]+\d+|\d+[a-zA-Z]+|[a-zA-Z]+_\d+)\b'
        matches = re.findall(pattern, user_input)
        return matches[0] if matches else None

    def _format_lead_scoring_response(self, prediction: dict) -> str:
        """Format lead scoring prediction into readable response"""
        response = f"📊 **Lead Scoring Result**\n\n"
        response += f"**Prospect ID:** {prediction['prospect_id']}\n"
        response += f"**Predicted Score:** {prediction['predicted_score']:.2f}/5.0\n\n"
        
        # Priority level
        score = prediction['predicted_score']
        if score >= 4.0:
            priority = "🔥 **URGENT PRIORITY**"
        elif score >= 3.0:
            priority = "⚡ **HIGH PRIORITY**"
        elif score >= 2.0:
            priority = "📈 **MEDIUM PRIORITY**"
        else:
            priority = "📋 **LOW PRIORITY**"
        
        response += f"**Priority Level:** {priority}\n\n"
        
        # Input data summary
        input_data = prediction['input_data']
        response += "**Lead Details:**\n"
        response += f"- Beat Tag: {input_data['beat_tag']}\n"
        response += f"- Turnover: {input_data['turnover']}\n"
        response += f"- Intent Score: {input_data['intent_score']}/100\n"
        response += f"- Profile Score: {input_data['profile_score']}/100\n"
        
        return response

# Enhanced creation functions with MySQL support

def create_lead_prioritization_assistant(use_mysql=True, db_config=None):
    """Create lead prioritization assistant with optional MySQL integration"""
    assistant = ContextAwareAssistant(use_mysql=use_mysql, db_config=db_config)
    
    assistant.create_system_prompt(
        domain_context="lead prioritization and scoring with beat plan tagging and MySQL integration",
        personality="analytical and data-driven",
        expertise_areas=[
            "buyer behavior analysis", 
            "engagement history scoring", 
            "lead source evaluation",
            "conversion probability",
            "sales pipeline optimization",
            "beat plan tag analysis",
            "account value assessment",
            "MySQL database integration"
        ]
    )
    
    return assistant

def create_support_assistant():
    assistant = ContextAwareAssistant()
    
    assistant.create_system_prompt(
        domain_context="customer support",
        personality="friendly and solution-oriented",
        expertise_areas=["product troubleshooting", "billing", "technical support"]
    )
    
    return assistant

def create_code_assistant():
    assistant = ContextAwareAssistant()
    
    assistant.create_system_prompt(
        domain_context="software development",
        personality="technical and precise",
        expertise_areas=["Python", "JavaScript", "debugging", "architecture"]
    )
    
    return assistant

# Enhanced standalone function with MySQL support
def generate_response(user_input, recent_messages=None, assistant_type="lead_priority", use_context=True, use_mysql=True):
    """
    Enhanced function to generate responses with MySQL integration
    NOTE: Changed default to lead_priority to enable database access
    """
    if assistant_type == "code":
        assistant = create_code_assistant()
    elif assistant_type == "support":
        assistant = create_support_assistant()
    elif assistant_type == "lead_priority":
        assistant = create_lead_prioritization_assistant(use_mysql=use_mysql)
    else:
        assistant = ContextAwareAssistant(use_mysql=use_mysql)
    
    return assistant.generate_response(user_input, recent_messages, use_context)

# Direct MySQL functions for quick access
def train_lead_scoring_model():
    """Quick function to train the lead scoring model"""
    try:
        results = quick_train_and_predict()
        return f"Model trained successfully. Generated predictions for {len(results)} leads."
    except Exception as e:
        return f"Training failed: {str(e)}"

def score_prospect(prospect_id: str):
    """Quick function to score a specific prospect"""
    try:
        result = quick_train_and_predict(prospect_id=prospect_id)
        return result
    except Exception as e:
        return f"Scoring failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example 1: Using the enhanced assistant with MySQL
    assistant = create_lead_prioritization_assistant(use_mysql=True)
    
    # Test various queries
    test_queries = [
        "train model",
        "score lead abc123",
        "show me top 5 leads",
        "leads with ERV win back beat tag",
        "leads from Mumbai",
        "leads with high intent scores"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        response = assistant.generate_response(query)
        print(response)