# utils.py - Enhanced with MySQL Integration, Comprehensive Lead Scoring, and CSV Download

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import io
import csv
import platform
from pathlib import Path

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

class CSVExportManager:
    """Manages CSV export functionality for lead data"""
    
    def __init__(self):
        self.export_directory = self.get_downloads_folder()
        self.ensure_export_directory()       
    

    def get_downloads_folder(self):
        """Get the system's Downloads folder path"""
        try:
            # For Windows
            if platform.system() == "Windows":
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders") as key:
                    downloads_path = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")[0]
                    return downloads_path
            
            # For macOS and Linux
            else:
                home = Path.home()
                downloads_path = home / "Downloads"
                return str(downloads_path)
                
        except Exception as e:
            print(f"Could not find Downloads folder: {e}")
            # Fallback to user home directory
            return str(Path.home() / "Downloads")
    
    def ensure_export_directory(self):
        """Ensure export directory exists"""
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)
    
    def export_leads_to_csv(self, leads_data: List[Dict], filename: str = None) -> str:
        """Export leads data to CSV file"""
        if not leads_data:
            return None
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leads_export_{timestamp}.csv"
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = os.path.join(self.export_directory, filename)
        
        # Convert to DataFrame and export
        df = pd.DataFrame(leads_data)
        
        # Clean and format the data for better CSV output
        df = self.clean_dataframe_for_export(df)
        
        # Export to CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
    
    def clean_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format DataFrame for CSV export"""
        # Flatten nested dictionaries
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Check if column contains dictionaries
                    if any(isinstance(x, dict) for x in df[col].dropna()):
                        # Flatten dictionary columns
                        for idx, value in df[col].items():
                            if isinstance(value, dict):
                                df.loc[idx, col] = str(value)
                except:
                    pass
        
        # Rename columns to be more user-friendly
        column_mapping = {
            'prospectId': 'Prospect_ID',
            'beat_tag': 'Beat_Tag',
            'turnover': 'Annual_Turnover_Cr',
            'lead_intent_score': 'Intent_Score',
            'lead_profile_score': 'Profile_Score',
            'address': 'Address',
            'predicted_score': 'AI_Predicted_Score',
            'priority_level': 'Priority_Level'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add timestamp
        df['Export_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return df
    
    def create_csv_download_response(self, leads_data: List[Dict], query_type: str, criteria: Dict = None) -> Dict:
        """Create response with CSV download information"""
        if not leads_data:
            return {
                "status": "error",
                "message": "No data to export",
                "csv_path": None
            }
        
        # Generate descriptive filename
        query_clean = query_type.lower().replace(' ', '_').replace('leads', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{query_clean}_leads_{timestamp}.csv"
        
        # Export to CSV
        csv_path = self.export_leads_to_csv(leads_data, filename)
        
        if csv_path:
            return {
                "status": "success",
                "message": f"CSV file created: {filename}",
                "csv_path": csv_path,
                "record_count": len(leads_data),
                "filename": filename
            }
        else:
            return {
                "status": "error",
                "message": "Failed to create CSV file",
                "csv_path": None
            }

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
        elif 'meeting_days_from_today' in lead_data:
            # Use days from today to create a date
            future_date = datetime.now() + timedelta(days=lead_data['meeting_days_from_today'])
            scores['meeting_score'] = self.calculate_meeting_score(
                future_date.strftime('%Y-%m-%d')
            )
        
        # 2. Purchase Cycle Date Score
        if 'purchase_cycle_date' in lead_data:
            scores['purchase_cycle_score'] = self.calculate_purchase_cycle_score(
                lead_data['purchase_cycle_date']
            )
        elif 'purchase_cycle_days_from_today' in lead_data:
            future_date = datetime.now() + timedelta(days=lead_data['purchase_cycle_days_from_today'])
            scores['purchase_cycle_score'] = self.calculate_purchase_cycle_score(
                future_date.strftime('%Y-%m-%d')
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
        elif 'days_since_last_meeting' in lead_data:
            past_date = datetime.now() - timedelta(days=lead_data['days_since_last_meeting'])
            scores['recency_score'] = self.calculate_last_meeting_recency_score(
                past_date.strftime('%Y-%m-%d')
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
            
            # Map score types to actual score keys
            score_mapping = {
                'meeting_scheduled_followup': 'meeting_score',
                'purchase_cycle_date': 'purchase_cycle_score',
                'lead_intent_score': 'intent_score',
                'last_meeting_date': 'recency_score',
                'minimum_meetings_gap': 'meetings_gap_score',
                'account_value_profitability': 'account_value_score',
                'beat_plan_tagging': 'beat_plan_score'
            }
            
            actual_score_key = score_mapping.get(score_type, score_key)
            
            if actual_score_key in scores and 'logic_score' in scores[actual_score_key]:
                raw_score = scores[actual_score_key]['logic_score']
                # For last meeting date, we want recent to be better (positive contribution)
                if score_type == 'last_meeting_date':
                    weighted_score = raw_score * abs(weight)  # Use positive weight
                else:
                    weighted_score = raw_score * weight
                    
                final_score += weighted_score
                score_breakdown[score_type] = {
                    'raw_score': raw_score,
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
        
        # Initialize lead scoring engine
        self.lead_scorer = LeadScoringEngine()
        
        # Initialize CSV export manager
        self.csv_manager = CSVExportManager()
        
        # MySQL integration
        self.use_mysql = use_mysql
        self.mysql_trainer = None
        self.model_trained = False
        
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
                results = self.mysql_trainer.train_from_database(limit=100)
                self.model_trained = True
                print("✅ Model auto-trained successfully")
                return True
            except Exception as e:
                print(f"❌ Auto-training failed: {e}")
                return False
        return self.model_trained

    def create_system_prompt(self, domain_context="", personality="helpful", expertise_areas=None):
        """Create customized system prompt for specific use cases"""
        base_prompt = f"You are a {personality} assistant"
        
        if domain_context:
            base_prompt += f" specialized in {domain_context}"
        
        if expertise_areas:
            expertise_list = ", ".join(expertise_areas)
            base_prompt += f" with expertise in: {expertise_list}"
        
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

CSV EXPORT CAPABILITY:
- For lead queries that return multiple results, you can offer CSV download
- Users can export filtered lead lists for external analysis
- CSV files include all relevant lead data with user-friendly column names

You can score leads, provide rankings, suggest prioritization strategies, and export data to CSV using real database data.
            """
        
        return base_prompt + """
        
Instructions:
- Provide accurate, helpful responses based on the context provided
- If you don't know something, say so clearly
- Use the conversation history to maintain context
- Be concise but thorough in your explanations
- For lead scoring queries, use the MySQL database when available
- Use comprehensive scoring logic for new lead evaluations
- Offer CSV export for multi-lead results when appropriate
        """

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
            
            query = "SELECT * FROM nx_op_ld_ai_lead360"
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            query += " LIMIT 50"  # Increased limit for CSV export
            
            print(f"🔍 Debug Query: {query}")
            print(f"🔍 Debug Params: {params}")
            
            import pandas as pd
            results_df = pd.read_sql(query, self.mysql_trainer.db_manager.engine, params=params)
            
            return {
                "status": "success",
                "leads": results_df.to_dict('records'),
                "count": len(results_df)
            }
            
        except Exception as e:
            return {"error": f"Database query failed: {str(e)}"}

    def generate_response(self, user_input, history=None, use_context=True):
        """Generate response with context awareness, MySQL lead scoring, and CSV export"""
        try:
            # PRIORITY 1: Handle MySQL-specific commands first
            if self.use_mysql and any(keyword in user_input.lower() for keyword in 
                                     ['train model', 'score lead', 'prospect', 'top leads', 'database']):
                return self._handle_mysql_commands(user_input)
            
            # PRIORITY 2: Handle lead-related queries with database integration and CSV export
            if self.use_mysql and any(keyword in user_input.lower() for keyword in 
                                    ['lead', 'priority', 'rank', 'beat', 'turnover', 'intent', 'active leads', 'csv', 'download', 'export']):
                
                # Check for CSV export requests
                export_requested = any(term in user_input.lower() for term in ['csv', 'download', 'export', 'file'])
                
                # Try to extract prospect ID first
                prospect_id = self._extract_prospect_id(user_input)
                if prospect_id and len(prospect_id) >= 5:
                    result = self.score_lead_from_database(prospect_id)
                    if result.get("status") == "success":
                        return self._format_lead_scoring_response(result["prediction"])
                
                # Check if this is a new lead description
                if any(term in user_input.lower() for term in ['score this', 'new lead', 'score a', 'rate this', 'analyze this']):
                    return self._handle_new_lead_scoring(user_input)
                
                # Try to extract filtering criteria for existing leads
                criteria = self._extract_lead_criteria(user_input)
                if criteria or any(term in user_input.lower() for term in ['active leads', 'all leads']):
                    result = self.get_database_leads_by_criteria(criteria)
                    if result.get("status") == "success":
                        return self._format_leads_list_response(result["leads"], criteria, export_csv=export_requested)
                
                # For general lead queries, get top leads
                if any(term in user_input.lower() for term in ['best', 'top', 'highest', 'priority']):
                    result = self.get_top_leads(20)  # Increased for CSV export
                    if result.get("status") == "success":
                        return self._format_top_leads_response(result["top_leads"], export_csv=export_requested)

            # PRIORITY 3: Regular LLM response for non-lead queries
            return self._generate_regular_response(user_input, history, use_context)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg

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
                if hasattr(msg, 'sender'):
                    if msg.sender == 'human':
                        messages.append(HumanMessage(content=msg.text))
                    elif msg.sender == 'ai':
                        messages.append(AIMessage(content=msg.text))
                elif isinstance(msg, (HumanMessage, AIMessage)):
                    messages.append(msg)
                elif isinstance(msg, dict):
                    if msg.get('role') == 'user' or msg.get('type') == 'human':
                        messages.append(HumanMessage(content=msg.get('content', msg.get('text', ''))))
                    elif msg.get('role') == 'assistant' or msg.get('type') == 'ai':
                        messages.append(AIMessage(content=msg.get('content', msg.get('text', ''))))
        
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        # Generate response using the LLM
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
            import re
            numbers = re.findall(r'\d+', user_input)
            limit = int(numbers[0]) if numbers else 10
            
            # Check for CSV export request
            export_csv = any(term in user_input_lower for term in ['csv', 'download', 'export'])
            
            result = self.get_top_leads(limit)
            if result.get("status") == "success":
                leads = result["top_leads"]
                return self._format_top_leads_response(leads, export_csv=export_csv)
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
        
        # Look for patterns like abc123, but exclude business terms
        patterns = [
            r'\b([a-zA-Z]{3,}\d{3,})\b',  # abc123, LEAD001
            r'\b([a-zA-Z]+_\d+)\b',       # LEAD_001
        ]
        
        exclude_terms = ['100cr', '500cr', '50cr', '10cr', '1000cr', '25cr']
        
        for pattern in patterns:
            matches = re.findall(pattern, user_input.lower())
            for match in matches:
                if not any(exclude in match.lower() for exclude in exclude_terms):
                    return match
        
        # Fallback for exact database patterns
        exact_pattern = r'\b(abc\d{3})\b'
        exact_match = re.search(exact_pattern, user_input.lower())
        if exact_match:
            return exact_match.group(1)
        
        return None

    def _extract_lead_criteria(self, user_input: str) -> dict:
        """Extract lead filtering criteria from user input"""
        criteria = {}
        user_input_lower = user_input.lower()
        
        # Extract beat tag
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
        
        # Extract city
        cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'pune', 'hyderabad']
        for city in cities:
            if city in user_input_lower:
                criteria['city'] = city.title()
                break
        
        return criteria

    def _handle_new_lead_scoring(self, user_input: str) -> str:
        """Handle scoring of new lead descriptions using comprehensive scoring logic"""
        try:
            lead_data = self._extract_new_lead_data(user_input)
            
            if not lead_data:
                return "❌ Could not extract enough lead information. Please provide beat tag, turnover, and intent score."
            
            # Use comprehensive scoring logic instead of ML model for new leads
            scoring_result = self.lead_scorer.calculate_comprehensive_lead_score(lead_data)
            
            response = f"📊 **New Lead Comprehensive Scoring Result**\n\n"
            response += f"**Final Ranking Score:** {scoring_result['final_ranking_score']:.2f}/5.0\n\n"
            
            # Priority level based on comprehensive score
            score = scoring_result['final_ranking_score']
            if score >= 4.0:
                priority = "🔥 **URGENT PRIORITY**"
            elif score >= 3.0:
                priority = "⚡ **HIGH PRIORITY**"
            elif score >= 2.0:
                priority = "📈 **MEDIUM PRIORITY**"
            else:
                priority = "📋 **LOW PRIORITY**"
            
            response += f"**Priority Level:** {priority}\n\n"
            
            # Show detailed score breakdown
            response += "**Score Breakdown:**\n"
            for param, details in scoring_result['score_breakdown'].items():
                response += f"- {param.replace('_', ' ').title()}: {details['raw_score']:.2f} × {details['weight']} = {details['weighted_score']:.2f}\n"
            
            response += "\n**Individual Score Details:**\n"
            for score_type, score_data in scoring_result['individual_scores'].items():
                if 'description' in score_data:
                    response += f"- {score_data['description']}: {score_data.get('logic_score', 'N/A'):.2f}\n"
            
            response += "\n**Extracted Lead Data:**\n"
            key_fields = ['beat_plan_tag', 'annual_turnover', 'intent_score', 'meeting_days_from_today', 'purchase_cycle_days_from_today']
            for key in key_fields:
                if key in lead_data:
                    response += f"- {key.replace('_', ' ').title()}: {lead_data[key]}\n"
            
            # Add recommendations
            response += f"\n**Recommendation:** "
            if score >= 3.5:
                response += "Schedule immediate follow-up call within 24 hours"
            elif score >= 2.5:
                response += "Add to high-priority pipeline, contact within 3 days"
            elif score >= 1.5:
                response += "Standard nurturing sequence, follow up within a week"
            else:
                response += "Long-term nurturing or additional qualification needed"
            
            return response
            
        except Exception as e:
            return f"❌ Error scoring new lead: {str(e)}"

    def _extract_new_lead_data(self, user_input: str) -> dict:
        """Extract lead data from new lead description"""
        lead_data = {}
        user_input_lower = user_input.lower()
        
        # Extract beat tag
        beat_tag_mappings = {
            'erv win back': 'ERV_Winback',
            'win back': 'ERV_Winback', 
            'active': 'ERV_Active',
            'new lead': 'NCA_Hot',
            'hot': 'NCA_Hot',
            'warm': 'NCA_Warm',
            'cold': 'NCA_Cold',
            'dormant': 'ERV_Dormant'
        }
        
        for search_term, standard_value in beat_tag_mappings.items():
            if search_term in user_input_lower:
                lead_data['beat_plan_tag'] = standard_value
                break
        
        # Extract turnover with better pattern matching
        import re
        turnover_patterns = [
            (r'(\d+)\s*cr.*?(\d+)\s*cr', lambda m: int((int(m.group(1)) + int(m.group(2))) / 2)),  # Range like 100cr-500cr
            (r'(\d+)\s*crore', lambda m: int(m.group(1))),
            (r'rs\.?\s*(\d+)\s*cr', lambda m: int(m.group(1))),
            (r'(\d+)\s*cr', lambda m: int(m.group(1))),
            (r'turnover.*?(\d+)', lambda m: int(m.group(1)))
        ]
        
        for pattern, converter in turnover_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                lead_data['annual_turnover'] = converter(match)
                break
        
        # Extract intent score
        intent_patterns = [
            r'intent.*?(\d+)',
            r'(\d+).*?intent',
            r'engagement.*?(\d+)',
            r'interest.*?(\d+)'
        ]
        
        for pattern in intent_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                score = int(match.group(1))
                if score <= 100:  # Valid intent score
                    lead_data['intent_score'] = score
                    break
        
        # Extract profile score
        profile_patterns = [
            r'profile.*?(\d+)',
            r'(\d+).*?profile'
        ]
        
        for pattern in profile_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                score = int(match.group(1))
                if score <= 100:
                    lead_data['lead_profile_score'] = score
                    break
        
        # Extract meeting timeline
        meeting_patterns = [
            (r'meeting.*?(\d+)\s*days?', lambda m: int(m.group(1))),
            (r'(\d+)\s*days?.*?meeting', lambda m: int(m.group(1))),
            (r'next\s*week', lambda m: 7),
            (r'tomorrow', lambda m: 1),
            (r'today', lambda m: 0)
        ]
        
        for pattern, converter in meeting_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                lead_data['meeting_days_from_today'] = converter(match)
                break
        
        # Set intelligent defaults based on beat tag
        if 'beat_plan_tag' not in lead_data:
            lead_data['beat_plan_tag'] = 'NCA_Warm'
        
        if 'annual_turnover' not in lead_data:
            lead_data['annual_turnover'] = 100  # Default to medium value
        
        if 'intent_score' not in lead_data:
            # Set default based on beat tag
            intent_defaults = {
                'ERV_Winback': 70,
                'NCA_Hot': 80,
                'ERV_Active': 60,
                'NCA_Warm': 50,
                'NCA_Cold': 30,
                'ERV_Dormant': 25
            }
            lead_data['intent_score'] = intent_defaults.get(lead_data['beat_plan_tag'], 50)
        
        # Add derived fields for comprehensive scoring
        beat_tag = lead_data['beat_plan_tag']
        
        # Set meeting timeline based on beat tag if not specified
        if 'meeting_days_from_today' not in lead_data:
            meeting_defaults = {
                'ERV_Winback': 5,   # Urgent for win-back
                'NCA_Hot': 7,       # Quick for hot leads
                'ERV_Active': 14,   # Standard for active
                'NCA_Warm': 21,     # Longer for warm
                'NCA_Cold': 30,     # Even longer for cold
                'ERV_Dormant': 45   # Longest for dormant
            }
            lead_data['meeting_days_from_today'] = meeting_defaults.get(beat_tag, 21)
        
        # Set purchase cycle based on company size and beat tag
        if 'purchase_cycle_days_from_today' not in lead_data:
            base_cycle = 90
            if lead_data['annual_turnover'] >= 500:
                base_cycle = 120  # Longer for large companies
            elif lead_data['annual_turnover'] < 50:
                base_cycle = 60   # Shorter for small companies
            
            # Adjust based on beat tag
            cycle_adjustments = {
                'ERV_Winback': -30,  # Faster for win-back
                'NCA_Hot': -15,      # Faster for hot
                'ERV_Dormant': 45    # Slower for dormant
            }
            adjustment = cycle_adjustments.get(beat_tag, 0)
            lead_data['purchase_cycle_days_from_today'] = max(30, base_cycle + adjustment)
        
        # Set last meeting recency
        if 'days_since_last_meeting' not in lead_data:
            recency_defaults = {
                'ERV_Winback': 30,   # Had previous relationship
                'ERV_Active': 7,     # Recent activity
                'ERV_Dormant': 90,   # Long time ago
                'NCA_Hot': 3,        # Very recent inquiry
                'NCA_Warm': 14,      # Some recent activity
                'NCA_Cold': 60       # Older inquiry
            }
            lead_data['days_since_last_meeting'] = recency_defaults.get(beat_tag, 21)
        
        # Set meeting completion data
        target_meetings_map = {
            'ERV_Winback': 4,
            'NCA_Hot': 3,
            'ERV_Active': 3,
            'NCA_Warm': 2,
            'NCA_Cold': 2,
            'ERV_Dormant': 1
        }
        
        lead_data['target_meetings'] = target_meetings_map.get(beat_tag, 3)
        
        # Completed meetings based on engagement level
        if lead_data['intent_score'] >= 80:
            completion_rate = 0.8
        elif lead_data['intent_score'] >= 60:
            completion_rate = 0.6
        elif lead_data['intent_score'] >= 40:
            completion_rate = 0.4
        else:
            completion_rate = 0.2
        
        lead_data['completed_meetings'] = max(0, int(lead_data['target_meetings'] * completion_rate))
        
        # Add other required fields
        lead_data.update({
            'company_size': 'large' if lead_data['annual_turnover'] >= 500 else 'medium' if lead_data['annual_turnover'] >= 100 else 'small',
            'industry': 'manufacturing',
            'lead_source': 'manual_input',
            'engagement_level': 'high' if lead_data['intent_score'] >= 70 else 'medium' if lead_data['intent_score'] >= 50 else 'low'
        })
        
        return lead_data

    def _format_lead_scoring_response(self, prediction: dict) -> str:
        """Format lead scoring prediction into readable response"""
        response = f"📊 **Lead Scoring Result**\n\n"
        response += f"**Prospect ID:** {prediction['prospect_id']}\n"
        response += f"**Predicted Score:** {prediction['predicted_score']:.2f}/5.0\n\n"
        
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
        
        input_data = prediction['input_data']
        response += "**Lead Details:**\n"
        response += f"- Beat Tag: {input_data['beat_tag']}\n"
        response += f"- Turnover: {input_data['turnover']}\n"
        response += f"- Intent Score: {input_data['intent_score']}/100\n"
        response += f"- Profile Score: {input_data['profile_score']}/100\n"
        
        return response

    def _format_leads_list_response(self, leads: list, criteria: dict, export_csv: bool = False) -> str:
        """Format leads list response with optional CSV export"""
        if not leads:
            return f"❌ No leads found matching criteria: {criteria}"
        
        criteria_text = []
        for key, value in criteria.items():
            criteria_text.append(f"{key}: {value}")
        
        response = f"📋 **Found {len(leads)} leads"
        if criteria_text:
            response += f" matching criteria ({', '.join(criteria_text)})"
        response += ":**\n\n"
        
        # Show preview of leads (first 5)
        for i, lead in enumerate(leads[:5], 1):
            response += f"{i}. **{lead['prospectId']}**\n"
            response += f"   Beat Tag: {lead['beat_tag']}\n"
            response += f"   Turnover: {lead['turnover']}\n"
            response += f"   Intent Score: {lead['lead_intent_score']}/100\n"
            response += f"   Profile Score: {lead['lead_profile_score']}/100\n"
            response += f"   Location: {lead['address'][:50]}...\n\n"
        
        if len(leads) > 5:
            response += f"... and {len(leads) - 5} more leads\n\n"
        
        # Handle CSV export
        if export_csv or len(leads) > 10:
            try:
                query_type = "filtered_leads"
                if criteria:
                    query_type = "_".join([f"{k}_{v}" for k, v in criteria.items()])
                
                csv_result = self.csv_manager.create_csv_download_response(leads, query_type, criteria)
                
                if csv_result["status"] == "success":
                    response += f"📊 **CSV Export Available:**\n"
                    response += f"- File: `{csv_result['filename']}`\n"
                    response += f"- Records: {csv_result['record_count']}\n"
                    response += f"- Location: `{csv_result['csv_path']}`\n\n"
                    response += "💡 *The CSV file has been generated and saved locally. You can find it in the downloads directory for further analysis.*"
                else:
                    response += f"❌ CSV Export Failed: {csv_result['message']}"
            except Exception as e:
                response += f"❌ CSV Export Error: {str(e)}"
        else:
            response += "\n💡 *For CSV export of these results, add 'CSV' or 'download' to your query.*"
        
        return response

    def _format_top_leads_response(self, leads: list, export_csv: bool = False) -> str:
        """Format top leads response with optional CSV export"""
        response = f"🏆 **Top {len(leads)} Leads by Predicted Score:**\n\n"
        
        # Add priority levels to leads for better CSV data
        enriched_leads = []
        for lead in leads:
            score = lead['predicted_score']
            if score >= 4.0:
                priority = "URGENT"
            elif score >= 3.0:
                priority = "HIGH"
            elif score >= 2.0:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            
            enriched_lead = lead.copy()
            enriched_lead['priority_level'] = priority
            enriched_leads.append(enriched_lead)
        
        # Show preview
        for i, lead in enumerate(leads[:10], 1):
            score = lead['predicted_score']
            priority_emoji = "🔥" if score >= 4.0 else "⚡" if score >= 3.0 else "📈" if score >= 2.0 else "📋"
            
            response += f"{i}. {priority_emoji} **{lead['prospect_id']}** - Score: {score:.2f}\n"
            response += f"   Beat Tag: {lead['input_data']['beat_tag']}\n"
            response += f"   Turnover: {lead['input_data']['turnover']}\n\n"
        
        # Handle CSV export
        if export_csv or len(leads) > 10:
            try:
                csv_result = self.csv_manager.create_csv_download_response(enriched_leads, "top_leads")
                
                if csv_result["status"] == "success":
                    response += f"📊 **CSV Export Available:**\n"
                    response += f"- File: `{csv_result['filename']}`\n"
                    response += f"- Records: {csv_result['record_count']}\n"
                    response += f"- Location: `{csv_result['csv_path']}`\n\n"
                    response += "💡 *The CSV file contains all leads with AI predicted scores and priority levels for your analysis.*"
                else:
                    response += f"❌ CSV Export Failed: {csv_result['message']}"
            except Exception as e:
                response += f"❌ CSV Export Error: {str(e)}"
        else:
            response += "\n💡 *For CSV export of these results, add 'CSV' or 'download' to your query.*"
        
        return response

    def get_relevant_context(self, query, k=3):
        """Retrieve relevant context based on query"""
        if not self.vectorstore:
            return ""
        
        relevant_docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        return context

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
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

    def score_lead(self, lead_data: Dict) -> Dict:
        """Score a lead using the enhanced scoring engine"""
        return self.lead_scorer.calculate_comprehensive_lead_score(lead_data)

    def export_leads_to_csv(self, query_type: str = "leads", limit: int = 50) -> str:
        """Export leads directly to CSV"""
        if not self.use_mysql or not self.mysql_trainer:
            return "❌ MySQL integration not enabled"
        
        try:
            # Get all leads for export
            query = f"SELECT * FROM nx_op_ld_ai_lead360 LIMIT {limit}"
            df = pd.read_sql(query, self.mysql_trainer.db_manager.engine)
            leads_data = df.to_dict('records')
            
            csv_result = self.csv_manager.create_csv_download_response(leads_data, query_type)
            
            if csv_result["status"] == "success":
                return f"✅ **CSV Export Successful:**\n- File: `{csv_result['filename']}`\n- Records: {csv_result['record_count']}\n- Location: `{csv_result['csv_path']}`"
            else:
                return f"❌ Export failed: {csv_result['message']}"
                
        except Exception as e:
            return f"❌ Export error: {str(e)}"

# Enhanced creation functions with MySQL support and CSV export
def create_lead_prioritization_assistant(use_mysql=True, db_config=None):
    """Create lead prioritization assistant with optional MySQL integration and CSV export"""
    assistant = ContextAwareAssistant(use_mysql=use_mysql, db_config=db_config)
    
    assistant.create_system_prompt(
        domain_context="lead prioritization and scoring with beat plan tagging, MySQL integration, and CSV export capabilities",
        personality="analytical and data-driven",
        expertise_areas=[
            "buyer behavior analysis", 
            "engagement history scoring", 
            "lead source evaluation",
            "conversion probability",
            "sales pipeline optimization",
            "beat plan tag analysis",
            "account value assessment",
            "MySQL database integration",
            "comprehensive lead scoring",
            "data export and analysis",
            "CSV file generation"
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

def generate_response(user_input, recent_messages=None, assistant_type="lead_priority", use_context=True, use_mysql=True):
    """Enhanced function to generate responses with MySQL integration and CSV export"""
    if assistant_type == "code":
        assistant = create_code_assistant()
    elif assistant_type == "support":
        assistant = create_support_assistant()
    elif assistant_type == "lead_priority":
        assistant = create_lead_prioritization_assistant(use_mysql=use_mysql)
    else:
        assistant = ContextAwareAssistant(use_mysql=use_mysql)
    
    return assistant.generate_response(user_input, recent_messages, use_context)

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

def export_active_leads_csv():
    """Quick function to export active leads to CSV"""
    try:
        assistant = create_lead_prioritization_assistant(use_mysql=True)
        return assistant.export_leads_to_csv("active_leads", limit=100)
    except Exception as e:
        return f"Export failed: {str(e)}"

if __name__ == "__main__":
    assistant = create_lead_prioritization_assistant(use_mysql=True)
    
    test_queries = [
        "train model",
        "score lead abc123", 
        "show me top 5 leads",
        "Active Leads",  # This should now trigger CSV export
        "leads with Active beat tag download CSV",
        "export all leads to CSV",
        "Score this new lead: Active beat tag, 100cr turnover, 85 intent score",
        "Rate this ERV winback lead with 500cr turnover and 90 intent score",
        "Analyze this hot lead: 200cr revenue, meeting next week, 75 intent score"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        response = assistant.generate_response(query)
        print(response)