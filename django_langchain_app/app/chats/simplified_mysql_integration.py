import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine, text
import pymysql
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv

# Import the existing training modules
from lead_scoring_training import LeadScoringTrainer, TrainingConfig

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 3306
    database: str = "llm"
    username: str = "llmuser"
    password: str = "llmuser_2025"
    charset: str = "utf8mb4"
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            database=os.getenv('DB_NAME'),
            username=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            charset=os.getenv('DB_CHARSET', 'utf8mb4')
        )

class SimplifiedMySQLLeadManager:
    """Simplified MySQL manager for nx_op_ld_ai_lead360 table"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.engine = None
        self.connect()
        
        # Beat tag mapping to standardized format
        self.beat_tag_mapping = {
            'ERV win back': 'ERV_Winback',
            'Active': 'ERV_Active', 
            'New Lead': 'NCA_Hot',
            'Dormant': 'ERV_Dormant'
        }
        
        # Turnover mapping to numerical values (in crores)
        self.turnover_mapping = {
            'Rs. 10cr - Rs.50cr': 30,
            'Rs. 50cr - Rs.100cr': 75,
            'Rs. 100cr - Rs.500cr': 300,
            'Rs. 500cr - Rs.1000cr': 750
        }
        
        # Constitution mapping to company size
        self.constitution_mapping = {
            'proprietorship': 'small',
            'partnership': 'medium', 
            'pvt ltd': 'large'
        }
        
    def connect(self):
        """Establish database connection"""
        try:
            connection_string = (
                f"mysql+pymysql://{self.db_config.username}:{self.db_config.password}@"
                f"{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
                f"?charset={self.db_config.charset}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection established successfully")
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def fetch_training_data(self, limit: int = None) -> pd.DataFrame:
        """Fetch and transform data from nx_op_ld_ai_lead360 table for training"""
        
        query = """
        SELECT 
            id,
            prospectId as lead_id,
            primary_number,
            address,
            turnover,
            beat_tag,
            lead_profile_score,
            lead_intent_score as intent_score,
            product_interested_in,
            pincode,
            constitution
        FROM nx_op_ld_ai_lead360
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Fetched {len(df)} records from nx_op_ld_ai_lead360")
            
            # Transform data for training
            df_transformed = self._transform_data_for_training(df)
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error fetching training data: {str(e)}")
            raise
    
    def _transform_data_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data to training format"""
        
        # Create a copy for transformation
        df_transformed = df.copy()
        
        # Map beat tags to standardized format
        df_transformed['beat_plan_tag'] = df_transformed['beat_tag'].map(
            self.beat_tag_mapping
        ).fillna('NCA_Warm')  # Default for unmapped values
        
        # Convert turnover to numerical values
        df_transformed['annual_turnover'] = df_transformed['turnover'].map(
            self.turnover_mapping
        ).fillna(30)  # Default value
        
        # Map constitution to company size
        df_transformed['company_size'] = df_transformed['constitution'].map(
            self.constitution_mapping
        ).fillna('medium')  # Default value
        
        # Extract industry from product interest (simplified)
        df_transformed['industry'] = df_transformed['product_interested_in'].apply(
            self._extract_industry_from_products
        )
        
        # Add derived features for training
        df_transformed['lead_source'] = 'crm'  # All leads from CRM
        df_transformed['engagement_level'] = df_transformed['lead_profile_score'].apply(
            self._categorize_engagement_level
        )
        
        # Create synthetic meeting and timeline data based on existing scores
        df_transformed = self._add_synthetic_timeline_data(df_transformed)
        
        # Calculate target score based on existing scores and beat tags
        df_transformed['target_score'] = df_transformed.apply(
            self._calculate_target_score, axis=1
        )
        
        # Add required columns for training with default/derived values
        df_transformed['target_meetings'] = df_transformed.apply(
            lambda row: self._get_target_meetings(row['company_size'], row['beat_plan_tag']), axis=1
        )
        
        return df_transformed
    
    def _extract_industry_from_products(self, products: str) -> str:
        """Extract industry category from product interests"""
        if pd.isna(products):
            return 'manufacturing'
        
        products_lower = products.lower()
        
        if any(term in products_lower for term in ['iron', 'steel', 'hr sheet', 'tmt']):
            return 'steel_manufacturing'
        elif any(term in products_lower for term in ['cement', 'concrete']):
            return 'construction'
        elif any(term in products_lower for term in ['wire', 'coil']):
            return 'wire_manufacturing'
        else:
            return 'manufacturing'
    
    def _categorize_engagement_level(self, profile_score: int) -> str:
        """Categorize engagement level based on profile score"""
        if pd.isna(profile_score):
            return 'medium'
        
        if profile_score >= 70:
            return 'high'
        elif profile_score >= 50:
            return 'medium'
        else:
            return 'low'
    
    def _add_synthetic_timeline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic timeline data based on existing scores"""
        
        # Generate meeting timeline based on engagement level
        df['meeting_days_from_today'] = df['engagement_level'].apply(
            lambda x: np.random.randint(-7, 14) if x == 'high' 
                     else np.random.randint(0, 30) if x == 'medium'
                     else np.random.randint(14, 60)
        )
        
        # Generate purchase cycle based on company size and beat tag
        df['purchase_cycle_days_from_today'] = df.apply(
            lambda row: self._estimate_purchase_cycle_days(row), axis=1
        )
        
        # Generate last meeting recency
        df['days_since_last_meeting'] = df['engagement_level'].apply(
            lambda x: np.random.randint(1, 7) if x == 'high'
                     else np.random.randint(7, 30) if x == 'medium'
                     else np.random.randint(30, 90)
        )
        
        # Generate completed meetings
        df['completed_meetings'] = df['engagement_level'].apply(
            lambda x: np.random.randint(2, 6) if x == 'high'
                     else np.random.randint(1, 4) if x == 'medium'
                     else np.random.randint(0, 3)
        )
        
        return df
    
    def _estimate_purchase_cycle_days(self, row) -> int:
        """Estimate purchase cycle days based on company profile"""
        base_days = 90
        
        # Adjust based on company size
        if row['company_size'] == 'large':
            base_days += 60
        elif row['company_size'] == 'small':
            base_days -= 30
        
        # Adjust based on beat tag
        if row['beat_plan_tag'] == 'ERV_Winback':
            base_days -= 30  # Faster for win-back
        elif row['beat_plan_tag'] == 'NCA_Hot':
            base_days -= 15  # Faster for hot leads
        elif row['beat_plan_tag'] == 'ERV_Dormant':
            base_days += 45  # Slower for dormant
        
        # Add some randomness
        return max(15, base_days + np.random.randint(-30, 31))
    
    def _calculate_target_score(self, row) -> float:
        """Calculate target score based on existing data"""
        base_score = 1.0
        
        # Score based on lead intent score
        if row['intent_score'] >= 85:
            base_score += 1.5
        elif row['intent_score'] >= 70:
            base_score += 1.0
        elif row['intent_score'] >= 50:
            base_score += 0.5
        
        # Score based on lead profile score
        if row['lead_profile_score'] >= 70:
            base_score += 1.0
        elif row['lead_profile_score'] >= 50:
            base_score += 0.5
        
        # Score based on beat tag
        beat_scores = {
            'ERV_Winback': 1.5,
            'NCA_Hot': 1.2,
            'ERV_Active': 1.0,
            'NCA_Warm': 0.8,
            'ERV_Dormant': 0.4
        }
        base_score += beat_scores.get(row['beat_plan_tag'], 0.5)
        
        # Score based on annual turnover
        if row['annual_turnover'] >= 500:
            base_score += 1.0
        elif row['annual_turnover'] >= 100:
            base_score += 0.6
        elif row['annual_turnover'] >= 50:
            base_score += 0.3
        
        return min(5.0, base_score)  # Cap at 5.0
    
    def _get_target_meetings(self, company_size: str, beat_tag: str) -> int:
        """Get target meetings based on company size and beat tag"""
        base_meetings = {'small': 2, 'medium': 3, 'large': 4}
        
        meetings = base_meetings.get(company_size, 3)
        
        # Adjust based on beat tag
        if beat_tag in ['ERV_Winback', 'NCA_Hot']:
            meetings += 1
        elif beat_tag == 'ERV_Dormant':
            meetings = max(1, meetings - 1)
        
        return meetings
    
    def save_prediction_to_table(self, prospect_id: str, predicted_score: float, 
                                model_version: str = None):
        """Save prediction back to the table (you might want to add a predictions column)"""
        
        # Note: This would require adding a predictions column to your table
        # For now, just log the prediction
        logger.info(f"Prediction for {prospect_id}: {predicted_score:.2f}")
        
        # If you want to store predictions, you could create a separate predictions table
        # or add columns to the existing table
    
    def get_lead_by_prospect_id(self, prospect_id: str) -> Dict:
        """Get a single lead by prospect ID for prediction"""
        
        query = """
        SELECT * FROM nx_op_ld_ai_lead360 
        WHERE prospectId = %(prospect_id)s
        """
        
        try:
            df = pd.read_sql(query, self.engine, params={'prospect_id': prospect_id})
            
            if df.empty:
                raise ValueError(f"Lead {prospect_id} not found")
            
            # Transform for prediction
            df_transformed = self._transform_data_for_training(df)
            
            return df_transformed.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error fetching lead {prospect_id}: {str(e)}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

class SimplifiedMySQLLeadScoringTrainer(LeadScoringTrainer):
    """Simplified trainer for nx_op_ld_ai_lead360 table"""
    
    def __init__(self, db_config: DatabaseConfig, training_config: TrainingConfig = None):
        super().__init__(training_config or TrainingConfig())
        self.db_manager = SimplifiedMySQLLeadManager(db_config)
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def train_from_database(self, limit: int = None) -> dict:
        """Train models using data from nx_op_ld_ai_lead360 table"""
        
        logger.info("Starting training from nx_op_ld_ai_lead360 table...")
        
        # Fetch training data
        training_data = self.db_manager.fetch_training_data(limit=limit)
        
        if len(training_data) < 10:
            logger.warning(f"Very limited training data: {len(training_data)} records")
            
        logger.info(f"Training with {len(training_data)} records")
        
        # Train models
        results = self.train_models(training_data, optimize_hyperparameters=True)
        
        return results
    
    def predict_for_prospect(self, prospect_id: str, model_name: str = 'random_forest') -> dict:
        """Predict lead score for a specific prospect ID"""
        
        try:
            # Get lead data from database
            lead_data = self.db_manager.get_lead_by_prospect_id(prospect_id)
            
            # Make prediction
            prediction_result = self.predict_lead_score(lead_data, model_name)
            
            # Save prediction (optional - you might want to store this)
            self.db_manager.save_prediction_to_table(
                prospect_id, 
                prediction_result['predicted_score'],
                self.model_version
            )
            
            return {
                'prospect_id': prospect_id,
                'predicted_score': prediction_result['predicted_score'],
                'model_used': model_name,
                'input_data': {
                    'beat_tag': lead_data.get('beat_tag'),
                    'turnover': lead_data.get('turnover'),
                    'intent_score': lead_data.get('intent_score'),
                    'profile_score': lead_data.get('lead_profile_score')
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting for prospect {prospect_id}: {str(e)}")
            raise
    
    def batch_predict_all_leads(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """Predict scores for all leads in the table"""
        
        try:
            # Get all prospect IDs
            query = "SELECT prospectId FROM nx_op_ld_ai_lead360"
            prospect_ids_df = pd.read_sql(query, self.db_manager.engine)
            
            results = []
            
            for _, row in prospect_ids_df.iterrows():
                try:
                    prediction = self.predict_for_prospect(row['prospectId'], model_name)
                    results.append(prediction)
                except Exception as e:
                    logger.error(f"Error predicting for {row['prospectId']}: {str(e)}")
            
            results_df = pd.DataFrame(results)
            
            # Sort by predicted score (highest first)
            if not results_df.empty:
                results_df = results_df.sort_values('predicted_score', ascending=False)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise

# Usage functions for easy integration

def create_mysql_lead_scoring_assistant(db_config: DatabaseConfig = None):
    """Create lead scoring assistant with MySQL integration"""
    
    if db_config is None:
        db_config = DatabaseConfig.from_env()
    
    trainer = SimplifiedMySQLLeadScoringTrainer(db_config)
    return trainer

def quick_train_and_predict(prospect_id: str = None, db_config: DatabaseConfig = None):
    """Quick function to train model and predict"""
    
    # Initialize trainer
    trainer = create_mysql_lead_scoring_assistant(db_config)
    
    # Train the model
    logger.info("Training model...")
    training_results = trainer.train_from_database()
    
    logger.info("Training completed. Model performance:")
    for model_name, metrics in training_results.items():
        logger.info(f"{model_name}: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.3f}")
    
    # Predict for specific prospect if provided
    if prospect_id:
        logger.info(f"Predicting for prospect {prospect_id}...")
        prediction = trainer.predict_for_prospect(prospect_id)
        logger.info(f"Prediction result: {prediction}")
        return prediction
    
    # Otherwise, predict for all leads
    logger.info("Predicting for all leads...")
    all_predictions = trainer.batch_predict_all_leads()
    logger.info(f"Generated predictions for {len(all_predictions)} leads")
    
    return all_predictions

# Example usage
if __name__ == "__main__":
    # Set up database configuration
    db_config = DatabaseConfig(
        host="localhost",
        database="your_database_name",
        username="root",
        password="your_password"
    )
    
    # Quick training and prediction
    try:
        # Train and predict for all leads
        results = quick_train_and_predict(db_config=db_config)
        
        # Show top 10 leads
        if not results.empty:
            print("\nTop 10 Predicted Leads:")
            print(results.head(10)[['prospect_id', 'predicted_score']].to_string(index=False))
        
        # Predict for specific prospect
        specific_prediction = quick_train_and_predict(
            prospect_id="abc123", 
            db_config=db_config
        )
        
        print(f"\nSpecific prediction: {specific_prediction}")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")