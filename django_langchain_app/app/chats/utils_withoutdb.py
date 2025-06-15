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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import mysql_integration

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
YOUR_GOOGLE_API_KEY = os.getenv('YOUR_GOOGLE_API_KEY')

class LeadScoringEngine:
    """Enhanced lead scoring engine with beat tags and ranking logic"""
    
    def __init__(self):
        # Beat Plan Tag Definitions based on Excel data
        self.beat_plan_tags = {
            'ERV_Active': {
                'sub_tags': ['ERV Active with Credit', 'ERV Active without Credit'],
                'base_score': 1.5,
                'description': 'Early Revenue Verification - Active accounts'
            },
            'ERV_Winback': {
                'sub_tags': ['ERV Win Back with Credit', 'ERV Win Back without Credit'],
                'base_score': 2.0,
                'description': 'Early Revenue Verification - Win Back campaigns'
            },
            'ERV_Dormant': {
                'sub_tags': ['ERV Dormant with Credit', 'ERV Dormant without Credit'],
                'base_score': 1.0,
                'description': 'Early Revenue Verification - Dormant accounts'
            },
            'NCA_Hot': {
                'sub_tags': ['NCA with Inquiry in last week with Credit with KDM', 
                           'NCA with Inquiry in last week without Credit with KDM'],
                'base_score': 1.5,
                'description': 'New Customer Acquisition - Hot leads'
            },
            'NCA_Warm': {
                'sub_tags': ['NCA with Inquiry in last month with Credit with KDM',
                           'NCA with Inquiry in last month without Credit with KDM'],
                'base_score': 1.0,
                'description': 'New Customer Acquisition - Warm leads'
            },
            'NCA_Cold': {
                'sub_tags': ['NCA with Inquiry in last quarter with Credit with KDM',
                           'NCA with Inquiry in last quarter without Credit with KDM'],
                'base_score': 0.5,
                'description': 'New Customer Acquisition - Cold leads'
            }
        }
        
        # Scoring weights from Excel
        self.scoring_weights = {
            'meeting_scheduled_followup': 0.6,
            'purchase_cycle_date': 0.2,
            'lead_intent_score': 0.25,
            'last_meeting_date': -0.4,  # Negative because recent is better
            'minimum_meetings_gap': 0.1,
            'account_value_profitability': 0.1,
            'beat_plan_tagging': 0.95
        }
        
        # Account value tiers (in INR)
        self.account_value_tiers = {
            'tier_1': {'min': 0, 'max': 25, 'score': 0.4, 'label': 'INR <25 cr'},
            'tier_2': {'min': 25, 'max': 100, 'score': 0.6, 'label': 'INR 25-100 cr'},
            'tier_3': {'min': 100, 'max': 500, 'score': 0.8, 'label': 'INR 100-500 cr'},
            'tier_4': {'min': 500, 'max': float('inf'), 'score': 1.0, 'label': 'INR >500 cr'}
        }

    def calculate_meeting_score(self, meeting_date: str, followup_date: str = None) -> Dict:
        """Calculate meeting scheduling score"""
        try:
            meeting_dt = datetime.strptime(meeting_date, '%Y-%m-%d')
            today = datetime.now()
            
            # Days difference from today
            days_diff = (meeting_dt - today).days
            
            if days_diff <= -2:  # Overdue
                logic_score = 1
                value_range = "Value: -2 to 2"
            elif -2 < days_diff <= 7:  # Within a week
                logic_score = 0.2
                value_range = "Value: 3 to 7"
            elif 7 < days_diff <= 30:  # Within a month
                logic_score = 0.2
                value_range = "Value: -5 to -3"
            else:  # Future
                logic_score = 1
                value_range = "Value: -2 to 2"
            
            return {
                'logic_score': logic_score,
                'value_range': value_range,
                'days_difference': days_diff,
                'description': 'Most Recent Meeting Scheduled/Follow-Up Date'
            }
        except ValueError:
            return {'logic_score': 0, 'error': 'Invalid date format'}

    def calculate_purchase_cycle_score(self, purchase_cycle_date: str) -> Dict:
        """Calculate purchase cycle date score"""
        try:
            cycle_dt = datetime.strptime(purchase_cycle_date, '%Y-%m-%d')
            today = datetime.now()
            days_diff = (cycle_dt - today).days
            
            if days_diff <= 30:  # Within 30 days
                logic_score = 0.2
            elif 30 < days_diff <= 90:  # Within 90 days
                logic_score = 0.2
            else:  # Beyond 90 days
                logic_score = 0.2
            
            return {
                'logic_score': logic_score,
                'days_to_cycle': days_diff,
                'description': 'Purchase Cycle Date'
            }
        except ValueError:
            return {'logic_score': 0, 'error': 'Invalid date format'}

    def calculate_lead_intent_score(self, intent_score: float) -> Dict:
        """Calculate lead intent score (0-100 scale)"""
        normalized_score = min(max(intent_score / 100.0, 0), 1)
        
        return {
            'logic_score': normalized_score,
            'raw_score': intent_score,
            'description': 'Beat Plan Tagging based on engagement metrics'
        }

    def calculate_last_meeting_recency_score(self, last_meeting_date: str) -> Dict:
        """Calculate recency score for last meeting"""
        try:
            meeting_dt = datetime.strptime(last_meeting_date, '%Y-%m-%d')
            today = datetime.now()
            days_since = (today - meeting_dt).days
            
            if days_since <= 7:
                logic_score = 1
                value_range = "Value: 0 to 7"
            else:
                logic_score = 1  # Default for longer periods
                value_range = "Value: >7"
            
            return {
                'logic_score': logic_score,
                'days_since_meeting': days_since,
                'value_range': value_range,
                'description': 'Last Meeting Date'
            }
        except ValueError:
            return {'logic_score': 0, 'error': 'Invalid date format'}

    def calculate_meetings_gap_score(self, target_meetings: int, completed_meetings: int) -> Dict:
        """Calculate minimum meetings gap score"""
        if target_meetings == 0:
            return {'logic_score': 0, 'description': 'No target meetings set'}
        
        completion_ratio = completed_meetings / target_meetings
        gap_ratio = max(0, 1 - completion_ratio)
        
        return {
            'logic_score': gap_ratio,
            'target_meetings': target_meetings,
            'completed_meetings': completed_meetings,
            'gap_ratio': gap_ratio,
            'description': 'No. of times account met in last 30 days'
        }

    def calculate_account_value_score(self, annual_turnover: float) -> Dict:
        """Calculate account value/profitability score"""
        for tier_name, tier_info in self.account_value_tiers.items():
            if tier_info['min'] <= annual_turnover < tier_info['max']:
                return {
                    'logic_score': tier_info['score'],
                    'tier': tier_name,
                    'tier_label': tier_info['label'],
                    'annual_turnover': annual_turnover,
                    'description': 'Annual Turnover tier scoring'
                }
        
        return {'logic_score': 0, 'error': 'Unable to categorize turnover'}

    def get_beat_plan_tag_score(self, tag_name: str, sub_tag: str = None) -> Dict:
        """Get beat plan tag score"""
        if tag_name in self.beat_plan_tags:
            tag_info = self.beat_plan_tags[tag_name]
            return {
                'logic_score': tag_info['base_score'],
                'tag_name': tag_name,
                'sub_tag': sub_tag,
                'description': tag_info['description'],
                'available_sub_tags': tag_info['sub_tags']
            }
        return {'logic_score': 0, 'error': f'Unknown beat plan tag: {tag_name}'}

    def calculate_comprehensive_lead_score(self, lead_data: Dict) -> Dict:
        """Calculate comprehensive lead ranking score"""
        scores = {}
        
        # 1. Meeting Scheduled/Follow-up Score
        if 'meeting_date' in lead_data:
            scores['meeting_score'] = self.calculate_meeting_score(
                lead_data['meeting_date'], 
                lead_data.get('followup_date')
            )
        
        # 2. Purchase Cycle Date Score
        if 'purchase_cycle_date' in lead_data:
            scores['purchase_cycle_score'] = self.calculate_purchase_cycle_score(
                lead_data['purchase_cycle_date']
            )
        
        # 3. Lead Intent Score
        if 'intent_score' in lead_data:
            scores['intent_score'] = self.calculate_lead_intent_score(
                lead_data['intent_score']
            )
        
        # 4. Last Meeting Recency
        if 'last_meeting_date' in lead_data:
            scores['recency_score'] = self.calculate_last_meeting_recency_score(
                lead_data['last_meeting_date']
            )
        
        # 5. Meetings Gap Score
        if 'target_meetings' in lead_data and 'completed_meetings' in lead_data:
            scores['meetings_gap_score'] = self.calculate_meetings_gap_score(
                lead_data['target_meetings'],
                lead_data['completed_meetings']
            )
        
        # 6. Account Value Score
        if 'annual_turnover' in lead_data:
            scores['account_value_score'] = self.calculate_account_value_score(
                lead_data['annual_turnover']
            )
        
        # 7. Beat Plan Tag Score
        if 'beat_plan_tag' in lead_data:
            scores['beat_plan_score'] = self.get_beat_plan_tag_score(
                lead_data['beat_plan_tag'],
                lead_data.get('beat_plan_sub_tag')
            )
        
        # Calculate weighted final score
        final_score = 0
        score_breakdown = {}
        
        for score_type, weight in self.scoring_weights.items():
            score_key = score_type.replace('_', '_score').replace('score_score', '_score')
            if score_key in scores and 'logic_score' in scores[score_key]:
                weighted_score = scores[score_key]['logic_score'] * weight
                final_score += weighted_score
                score_breakdown[score_type] = {
                    'raw_score': scores[score_key]['logic_score'],
                    'weight': weight,
                    'weighted_score': weighted_score
                }
        
        return {
            'final_ranking_score': final_score,
            'individual_scores': scores,
            'score_breakdown': score_breakdown,
            'lead_data': lead_data
        }

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
        
        # Initialize lead scoring engine
        self.lead_scorer = LeadScoringEngine()
        
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
- ERV_Active: Early Revenue Verification - Active accounts (Score: 1.5)
- ERV_Winback: Early Revenue Verification - Win Back campaigns (Score: 2.0)  
- ERV_Dormant: Early Revenue Verification - Dormant accounts (Score: 1.0)
- NCA_Hot: New Customer Acquisition - Hot leads (Score: 1.5)
- NCA_Warm: New Customer Acquisition - Warm leads (Score: 1.0)
- NCA_Cold: New Customer Acquisition - Cold leads (Score: 0.5)

SCORING PARAMETERS:
1. Meeting Scheduled/Follow-Up Date (Weight: 0.6)
2. Purchase Cycle Date (Weight: 0.2)
3. Lead Intent Score (Weight: 0.25)
4. Last Meeting Date Recency (Weight: -0.4)
5. Minimum Meetings Gap (Weight: 0.1)
6. Account Value/Profitability (Weight: 0.1)
7. Beat Plan Tagging (Weight: 0.95)

ACCOUNT VALUE TIERS:
- INR <25 cr: Score 0.4
- INR 25-100 cr: Score 0.6
- INR 100-500 cr: Score 0.8
- INR >500 cr: Score 1.0
            """
        
        additional_instructions = """
        
Instructions:
- Provide accurate, helpful responses based on the context provided
- If you don't know something, say so clearly
- Use the conversation history to maintain context
- Be concise but thorough in your explanations
- For lead scoring queries, use the comprehensive scoring system described above
        """
        
        return base_prompt + additional_instructions

    def score_lead(self, lead_data: Dict) -> Dict:
        """Score a lead using the enhanced scoring engine"""
        return self.lead_scorer.calculate_comprehensive_lead_score(lead_data)

    def generate_response(self, user_input, history=None, use_context=True):
        """Generate response with context awareness and lead scoring capabilities"""
        
        # Check if this is a lead scoring request
        if any(keyword in user_input.lower() for keyword in ['score', 'lead', 'priority', 'rank']):
            # Try to extract lead data from the input
            lead_data = self.extract_lead_data_from_input(user_input)
            if lead_data:
                scoring_result = self.score_lead(lead_data)
                return self.format_lead_scoring_response(scoring_result)
        
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

    def extract_lead_data_from_input(self, user_input: str) -> Optional[Dict]:
        """Extract lead data from user input for scoring"""
        # This is a simplified extraction - in practice, you'd use NLP or structured input
        lead_data = {}
        
        # Look for common patterns in the input
        input_lower = user_input.lower()
        
        # Beat plan tag detection
        if 'erv active' in input_lower:
            lead_data['beat_plan_tag'] = 'ERV_Active'
        elif 'erv winback' in input_lower or 'win back' in input_lower:
            lead_data['beat_plan_tag'] = 'ERV_Winback'
        elif 'erv dormant' in input_lower:
            lead_data['beat_plan_tag'] = 'ERV_Dormant'
        elif 'nca hot' in input_lower or 'hot lead' in input_lower:
            lead_data['beat_plan_tag'] = 'NCA_Hot'
        elif 'nca warm' in input_lower or 'warm lead' in input_lower:
            lead_data['beat_plan_tag'] = 'NCA_Warm'
        elif 'nca cold' in input_lower or 'cold lead' in input_lower:
            lead_data['beat_plan_tag'] = 'NCA_Cold'
        
        # Simple pattern matching for other data points
        # In practice, you'd want more sophisticated extraction
        
        return lead_data if lead_data else None

    def format_lead_scoring_response(self, scoring_result: Dict) -> str:
        """Format the lead scoring result into a readable response"""
        final_score = scoring_result['final_ranking_score']
        
        response = f"**Lead Scoring Analysis**\n\n"
        response += f"**Final Ranking Score: {final_score:.2f}**\n\n"
        
        response += "**Score Breakdown:**\n"
        for param, details in scoring_result['score_breakdown'].items():
            response += f"- {param.replace('_', ' ').title()}: {details['raw_score']:.2f} × {details['weight']} = {details['weighted_score']:.2f}\n"
        
        response += "\n**Individual Score Details:**\n"
        for score_type, score_data in scoring_result['individual_scores'].items():
            if 'description' in score_data:
                response += f"- {score_data['description']}: {score_data.get('logic_score', 'N/A')}\n"
        
        # Priority recommendation
        if final_score >= 2.0:
            priority = "HIGH PRIORITY"
        elif final_score >= 1.0:
            priority = "MEDIUM PRIORITY"
        else:
            priority = "LOW PRIORITY"
        
        response += f"\n**Recommended Priority: {priority}**"
        
        return response

# Enhanced lead prioritization assistant
def create_lead_prioritization_assistant():
    #assistant = ContextAwareAssistant()
    assistant = MySQLLeadScoringTrainer()
    assistant.create_system_prompt(
        domain_context="lead prioritization and scoring with beat plan tagging",
        personality="analytical and data-driven",
        expertise_areas=[
            "buyer behavior analysis", 
            "engagement history scoring", 
            "lead source evaluation",
            "conversion probability",
            "sales pipeline optimization",
            "beat plan tag analysis",
            "account value assessment",
            "meeting recency scoring"
        ]
    )
    
    return assistant

# Example usage functions remain the same...
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
            "timestamp": time.time(),
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

# Prepare training data for OpenAI fine-tuning
class OpenAIFineTuningPreparator:
    def __init__(self):
        self.training_examples = []
    
    def create_training_examples(self, leads_data: List[Dict]) -> List[Dict]:
        """Convert lead data to OpenAI fine-tuning format"""
        examples = []
        
        for lead in leads_data:
            # Create natural language description
            prompt = self.create_lead_description(lead)
            
            # Calculate expected score
            expected_score = self.calculate_score(lead)
            
            # Format for OpenAI
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a lead scoring expert. Analyze the lead information and provide a score from 0.0 to 5.0 with detailed reasoning."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": f"Lead Score: {expected_score:.2f}\n\nReasoning:\n{self.generate_reasoning(lead, expected_score)}"
                    }
                ]
            }
            examples.append(example)
        
        return examples
    
    def create_lead_description(self, lead: Dict) -> str:
        """Convert structured lead data to natural language"""
        description = f"""
Lead Information:
- Beat Plan Tag: {lead.get('beat_plan_tag', 'Unknown')}
- Company Size: {lead.get('company_size', 'Unknown')}
- Industry: {lead.get('industry', 'Unknown')}
- Annual Turnover: ₹{lead.get('annual_turnover', 0)} crores
- Intent Score: {lead.get('intent_score', 0)}/100
- Meeting scheduled in {lead.get('meeting_days_from_today', 0)} days
- Purchase cycle expected in {lead.get('purchase_cycle_days_from_today', 0)} days
- Last meeting was {lead.get('days_since_last_meeting', 0)} days ago
- Completed {lead.get('completed_meetings', 0)} out of {lead.get('target_meetings', 0)} target meetings
- Lead Source: {lead.get('lead_source', 'Unknown')}
- Engagement Level: {lead.get('engagement_level', 'Unknown')}

Please analyze this lead and provide a comprehensive scoring with reasoning.
        """.strip()
        return description
    
    def generate_reasoning(self, lead: Dict, score: float) -> str:
        """Generate reasoning for the score"""
        reasoning_parts = []
        
        # Beat plan analysis
        beat_tag = lead.get('beat_plan_tag', '')
        if 'Hot' in beat_tag:
            reasoning_parts.append("• High priority beat plan tag indicates strong potential")
        elif 'Winback' in beat_tag:
            reasoning_parts.append("• Winback opportunity with previous relationship")
        elif 'Cold' in beat_tag:
            reasoning_parts.append("• Cold lead requires more nurturing")
        
        # Intent score analysis
        intent = lead.get('intent_score', 0)
        if intent >= 80:
            reasoning_parts.append("• Very high intent score shows strong buying signals")
        elif intent >= 60:
            reasoning_parts.append("• Good intent score indicates moderate interest")
        else:
            reasoning_parts.append("• Low intent score suggests early stage prospect")
        
        # Urgency analysis
        meeting_days = lead.get('meeting_days_from_today', 30)
        if meeting_days <= 7:
            reasoning_parts.append("• Upcoming meeting shows immediate opportunity")
        
        # Value analysis
        turnover = lead.get('annual_turnover', 0)
        if turnover >= 500:
            reasoning_parts.append("• High-value enterprise account")
        elif turnover >= 100:
            reasoning_parts.append("• Medium-value prospect with good potential")
        
        return "\n".join(reasoning_parts)

# Usage for OpenAI fine-tuning
preparator = OpenAIFineTuningPreparator()
training_examples = preparator.create_training_examples(your_leads_data)

# Save for OpenAI upload
import json
with open('lead_scoring_training.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')
        
def generate_response(user_input, recent_messages=None, assistant_type="code", use_context=True):
    """
    Standalone function to generate responses - wrapper around ContextAwareAssistant
    """
    if assistant_type == "code":
        assistant = create_code_assistant()
    elif assistant_type == "support":
        assistant = create_support_assistant()
    elif assistant_type == "lead_priority":
        assistant = create_lead_prioritization_assistant()
    else:
        assistant = ContextAwareAssistant()
    
    return assistant.generate_response(user_input, recent_messages, use_context)

# Example usage with enhanced lead scoring
if __name__ == "__main__":
    # Create lead scoring assistant
    lead_assistant = create_lead_prioritization_assistant()
    
    # Example lead data
    sample_lead = {
        'meeting_date': '2025-06-20',
        'purchase_cycle_date': '2025-07-15',
        'intent_score': 75,
        'last_meeting_date': '2025-06-10',
        'target_meetings': 4,
        'completed_meetings': 2,
        'annual_turnover': 150,  # INR crores
        'beat_plan_tag': 'NCA_Hot',
        'beat_plan_sub_tag': 'NCA with Inquiry in last week with Credit with KDM'
    }
    
    # Score the lead
    scoring_result = lead_assistant.score_lead(sample_lead)
    print("Lead Scoring Result:", json.dumps(scoring_result, indent=2, default=str))
    
    # Test conversational lead scoring
    response = lead_assistant.generate_response(
        "Score this NCA Hot lead: they have a meeting scheduled for next week, purchase cycle in July, intent score of 75, and annual turnover of 150 crores"
    )
    print("Conversational Response:", response)