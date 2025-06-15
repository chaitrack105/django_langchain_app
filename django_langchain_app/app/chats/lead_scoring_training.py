import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    model_types: List[str] = None
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['random_forest', 'gradient_boosting', 'linear_regression']

class LeadScoringTrainer:
    """Comprehensive training framework for lead scoring models"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.training_history = []
        
        # Initialize models
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            }
        }
    
    def generate_synthetic_training_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic training data based on Excel scoring logic"""
        np.random.seed(self.config.random_state)
        
        data = []
        
        # Beat plan tags and their probabilities
        beat_tags = {
            'ERV_Active': {'prob': 0.15, 'base_score': 1.5},
            'ERV_Winback': {'prob': 0.10, 'base_score': 2.0},
            'ERV_Dormant': {'prob': 0.15, 'base_score': 1.0},
            'NCA_Hot': {'prob': 0.20, 'base_score': 1.5},
            'NCA_Warm': {'prob': 0.25, 'base_score': 1.0},
            'NCA_Cold': {'prob': 0.15, 'base_score': 0.5}
        }
        
        for i in range(n_samples):
            # Generate random lead data
            lead = {}
            
            # Beat plan tag (categorical)
            beat_tag = np.random.choice(list(beat_tags.keys()), 
                                      p=[beat_tags[tag]['prob'] for tag in beat_tags.keys()])
            lead['beat_plan_tag'] = beat_tag
            
            # Meeting date (days from today: -30 to +60)
            meeting_days_diff = np.random.randint(-30, 61)
            lead['meeting_days_from_today'] = meeting_days_diff
            
            # Purchase cycle date (days from today: 0 to 365)
            purchase_days_diff = np.random.randint(0, 366)
            lead['purchase_cycle_days_from_today'] = purchase_days_diff
            
            # Lead intent score (0-100)
            lead['intent_score'] = np.random.randint(0, 101)
            
            # Last meeting recency (days ago: 1-90)
            lead['days_since_last_meeting'] = np.random.randint(1, 91)
            
            # Meetings data
            lead['target_meetings'] = np.random.randint(1, 11)
            lead['completed_meetings'] = np.random.randint(0, lead['target_meetings'] + 1)
            
            # Annual turnover (INR crores: 1-1000)
            lead['annual_turnover'] = np.random.exponential(100) + 1
            
            # Additional features that could influence scoring
            lead['company_size'] = np.random.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
            lead['industry'] = np.random.choice(['tech', 'finance', 'healthcare', 'manufacturing'], 
                                              p=[0.3, 0.25, 0.25, 0.2])
            lead['lead_source'] = np.random.choice(['website', 'linkedin', 'referral', 'cold_call'], 
                                                 p=[0.4, 0.3, 0.2, 0.1])
            lead['engagement_level'] = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.4, 0.3])
            
            # Calculate target score using the Excel logic
            target_score = self.calculate_target_score(lead, beat_tags)
            lead['target_score'] = target_score
            
            # Add some realistic noise/variance
            lead['actual_conversion'] = np.random.binomial(1, min(target_score / 5.0, 0.95))
            lead['deal_value'] = lead['annual_turnover'] * np.random.uniform(0.01, 0.1)
            
            data.append(lead)
        
        return pd.DataFrame(data)
    
    def calculate_target_score(self, lead: Dict, beat_tags: Dict) -> float:
        """Calculate target score based on Excel logic"""
        score = 0
        
        # Beat plan tag score (weight: 0.95)
        beat_score = beat_tags[lead['beat_plan_tag']]['base_score']
        score += beat_score * 0.95
        
        # Meeting scheduling score (weight: 0.6)
        meeting_days = lead['meeting_days_from_today']
        if meeting_days <= -2:  # Overdue
            meeting_score = 1
        elif -2 < meeting_days <= 7:  # Within a week
            meeting_score = 0.2
        elif 7 < meeting_days <= 30:  # Within a month
            meeting_score = 0.2
        else:  # Future
            meeting_score = 1
        score += meeting_score * 0.6
        
        # Purchase cycle score (weight: 0.2)
        purchase_days = lead['purchase_cycle_days_from_today']
        if purchase_days <= 30:
            purchase_score = 0.2
        elif purchase_days <= 90:
            purchase_score = 0.2
        else:
            purchase_score = 0.2
        score += purchase_score * 0.2
        
        # Intent score (weight: 0.25)
        intent_normalized = lead['intent_score'] / 100.0
        score += intent_normalized * 0.25
        
        # Last meeting recency (weight: -0.4, recent is better)
        days_since = lead['days_since_last_meeting']
        if days_since <= 7:
            recency_score = 1
        elif days_since <= 30:
            recency_score = 0.5
        else:
            recency_score = 0.1
        score += recency_score * 0.4  # Note: positive because recent is better
        
        # Meetings gap (weight: 0.1)
        if lead['target_meetings'] > 0:
            completion_ratio = lead['completed_meetings'] / lead['target_meetings']
            meetings_score = completion_ratio
        else:
            meetings_score = 0
        score += meetings_score * 0.1
        
        # Account value (weight: 0.1)
        turnover = lead['annual_turnover']
        if turnover < 25:
            value_score = 0.4
        elif turnover < 100:
            value_score = 0.6
        elif turnover < 500:
            value_score = 0.8
        else:
            value_score = 1.0
        score += value_score * 0.1
        
        return score
    
    def preprocess_data(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """Preprocess data for training"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['beat_plan_tag', 'company_size', 'industry', 'lead_source', 'engagement_level']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if fit_encoders:
                    self.encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.encoders[col].fit_transform(df_processed[col])
                else:
                    if col in self.encoders:
                        df_processed[f'{col}_encoded'] = self.encoders[col].transform(df_processed[col])
                
        # Create additional features
        df_processed['meetings_completion_rate'] = df_processed['completed_meetings'] / (df_processed['target_meetings'] + 1)
        df_processed['intent_score_normalized'] = df_processed['intent_score'] / 100.0
        df_processed['is_recent_meeting'] = (df_processed['meeting_days_from_today'] >= -7) & (df_processed['meeting_days_from_today'] <= 7)
        df_processed['is_high_value'] = df_processed['annual_turnover'] > 100
        df_processed['urgency_score'] = 1 / (df_processed['purchase_cycle_days_from_today'] + 1)
        
        # Feature engineering based on interactions
        df_processed['beat_intent_interaction'] = df_processed['beat_plan_tag_encoded'] * df_processed['intent_score_normalized']
        df_processed['value_urgency_interaction'] = df_processed['annual_turnover'] * df_processed['urgency_score']
        
        return df_processed
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        # Define feature columns
        feature_columns = [
            'beat_plan_tag_encoded', 'company_size_encoded', 'industry_encoded', 
            'lead_source_encoded', 'engagement_level_encoded',
            'meeting_days_from_today', 'purchase_cycle_days_from_today', 
            'intent_score', 'days_since_last_meeting',
            'target_meetings', 'completed_meetings', 'annual_turnover',
            'meetings_completion_rate', 'intent_score_normalized', 
            'is_recent_meeting', 'is_high_value', 'urgency_score',
            'beat_intent_interaction', 'value_urgency_interaction'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features]
        y = df['target_score']
        
        return X, y
    
    def train_models(self, df: pd.DataFrame, optimize_hyperparameters: bool = True) -> Dict:
        """Train multiple models and return performance metrics"""
        logger.info("Starting model training...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, fit_encoders=True)
        X, y = self.prepare_features_target(df_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        results = {}
        
        for model_name in self.config.model_types:
            logger.info(f"Training {model_name}...")
            
            model_config = self.model_configs[model_name]
            model = model_config['model']
            
            # Hyperparameter optimization
            if optimize_hyperparameters and model_config['params']:
                grid_search = GridSearchCV(
                    model, model_config['params'], 
                    cv=self.config.cv_folds, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=1
                )
                
                # Use scaled data for linear models, original for tree-based
                if model_name == 'linear_regression':
                    grid_search.fit(X_train_scaled, y_train)
                    y_pred = grid_search.predict(X_test_scaled)
                else:
                    grid_search.fit(X_train, y_train)
                    y_pred = grid_search.predict(X_test)
                
                best_model = grid_search.best_estimator_
                self.models[model_name] = best_model
                
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                
            else:
                # Train without hyperparameter optimization
                if model_name == 'linear_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                self.models[model_name] = model
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if model_name == 'linear_regression':
                cv_scores = cross_val_score(self.models[model_name], X_train_scaled, y_train, 
                                          cv=self.config.cv_folds, scoring='neg_mean_squared_error')
            else:
                cv_scores = cross_val_score(self.models[model_name], X_train, y_train, 
                                          cv=self.config.cv_folds, scoring='neg_mean_squared_error')
            
            results[model_name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_score_mean': -cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'feature_importance': self.get_feature_importance(model_name, X.columns)
            }
            
            logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Store training info
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'features_used': list(X.columns),
            'model_performance': results
        }
        self.training_history.append(training_record)
        
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict:
        """Get feature importance for the model"""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
        else:
            return {}
        
        feature_importance = dict(zip(feature_names, importance))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        self.feature_importance[model_name] = feature_importance
        return feature_importance
    
    def predict_lead_score(self, lead_data: Dict, model_name: str = 'random_forest') -> Dict:
        """Predict lead score for new data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        # Convert to DataFrame
        df = pd.DataFrame([lead_data])
        
        # Preprocess
        df_processed = self.preprocess_data(df, fit_encoders=False)
        X, _ = self.prepare_features_target(df_processed)
        
        # Scale if needed
        if model_name == 'linear_regression':
            X_scaled = self.scalers['standard'].transform(X)
            prediction = self.models[model_name].predict(X_scaled)[0]
        else:
            prediction = self.models[model_name].predict(X)[0]
        
        # Get feature contributions (for tree-based models)
        feature_contributions = {}
        if hasattr(self.models[model_name], 'feature_importances_'):
            for feature, importance in zip(X.columns, self.models[model_name].feature_importances_):
                feature_contributions[feature] = importance * X.iloc[0][feature]
        
        return {
            'predicted_score': prediction,
            'model_used': model_name,
            'confidence_interval': self.get_prediction_confidence(X, model_name),
            'feature_contributions': feature_contributions,
            'input_features': dict(X.iloc[0])
        }
    
    def get_prediction_confidence(self, X: pd.DataFrame, model_name: str) -> Tuple[float, float]:
        """Get prediction confidence interval (simplified)"""
        model = self.models[model_name]
        
        if hasattr(model, 'predict'):
            # For tree-based models with multiple estimators
            if hasattr(model, 'estimators_'):
                predictions = []
                for estimator in model.estimators_:
                    pred = estimator.predict(X)[0]
                    predictions.append(pred)
                
                std = np.std(predictions)
                mean_pred = np.mean(predictions)
                
                # 95% confidence interval
                lower = mean_pred - 1.96 * std
                upper = mean_pred + 1.96 * std
                
                return (lower, upper)
        
        # Default: return empty interval
        return (0, 0)
    
    def evaluate_model_ensemble(self, df_test: pd.DataFrame) -> Dict:
        """Evaluate ensemble of all trained models"""
        df_processed = self.preprocess_data(df_test, fit_encoders=False)
        X, y_true = self.prepare_features_target(df_processed)
        
        ensemble_predictions = []
        individual_predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'linear_regression':
                X_scaled = self.scalers['standard'].transform(X)
                y_pred = model.predict(X_scaled)
            else:
                y_pred = model.predict(X)
            
            individual_predictions[model_name] = y_pred
            ensemble_predictions.append(y_pred)
        
        # Simple ensemble: average predictions
        if ensemble_predictions:
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
            
            ensemble_mse = mean_squared_error(y_true, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
            ensemble_r2 = r2_score(y_true, ensemble_pred)
            
            return {
                'ensemble_performance': {
                    'mse': ensemble_mse,
                    'mae': ensemble_mae,
                    'r2': ensemble_r2
                },
                'individual_predictions': individual_predictions,
                'ensemble_predictions': ensemble_pred.tolist()
            }
        
        return {}
    
    def save_models(self, filepath_prefix: str):
        """Save trained models and preprocessing components"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = f"{filepath_prefix}_{model_name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save scalers and encoders
        preprocessing_path = f"{filepath_prefix}_preprocessing_{timestamp}.joblib"
        preprocessing_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance
        }
        joblib.dump(preprocessing_data, preprocessing_path)
        
        # Save training history
        history_path = f"{filepath_prefix}_training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        logger.info(f"Saved preprocessing components and history")
    
    def load_models(self, filepath_prefix: str, timestamp: str = None):
        """Load trained models and preprocessing components"""
        if timestamp is None:
            # Find latest timestamp
            import glob
            pattern = f"{filepath_prefix}_*_*.joblib"
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError("No model files found")
            
            # Extract timestamp from filename
            timestamps = []
            for file in files:
                parts = file.split('_')
                if len(parts) >= 3:
                    timestamps.append(parts[-1].replace('.joblib', ''))
            
            timestamp = max(timestamps)
        
        # Load models
        for model_name in self.config.model_types:
            model_path = f"{filepath_prefix}_{model_name}_{timestamp}.joblib"
            try:
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load preprocessing components
        preprocessing_path = f"{filepath_prefix}_preprocessing_{timestamp}.joblib"
        try:
            preprocessing_data = joblib.load(preprocessing_path)
            self.scalers = preprocessing_data['scalers']
            self.encoders = preprocessing_data['encoders']
            self.feature_importance = preprocessing_data['feature_importance']
            logger.info("Loaded preprocessing components")
        except FileNotFoundError:
            logger.warning(f"Preprocessing file not found: {preprocessing_path}")

class RealTimeTrainingSystem:
    """System for continuous learning from new data"""
    
    def __init__(self, trainer: LeadScoringTrainer):
        self.trainer = trainer
        self.feedback_data = []
        self.retrain_threshold = 100  # Retrain after N new samples
        
    def collect_feedback(self, lead_data: Dict, predicted_score: float, 
                        actual_score: float = None, conversion: bool = None):
        """Collect feedback for continuous learning"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'lead_data': lead_data,
            'predicted_score': predicted_score,
            'actual_score': actual_score,
            'conversion': conversion,
            'feedback_quality': self.calculate_feedback_quality(predicted_score, actual_score)
        }
        
        self.feedback_data.append(feedback)
        
        # Check if retrain is needed
        if len(self.feedback_data) >= self.retrain_threshold:
            self.trigger_retraining()
    
    def calculate_feedback_quality(self, predicted: float, actual: float) -> str:
        """Calculate quality of prediction"""
        if actual is None:
            return 'unknown'
        
        error = abs(predicted - actual)
        if error < 0.5:
            return 'excellent'
        elif error < 1.0:
            return 'good'
        elif error < 2.0:
            return 'fair'
        else:
            return 'poor'
    
    def trigger_retraining(self):
        """Trigger model retraining with new data"""
        logger.info("Triggering model retraining with feedback data...")
        
        # Convert feedback to training format
        feedback_df = self.prepare_feedback_for_training()
        
        if len(feedback_df) > 10:  # Minimum samples for retraining
            # Retrain models
            results = self.trainer.train_models(feedback_df, optimize_hyperparameters=False)
            logger.info(f"Retraining completed. New performance: {results}")
            
            # Clear feedback data
            self.feedback_data = []
    
    def prepare_feedback_for_training(self) -> pd.DataFrame:
        """Convert feedback data to training format"""
        training_data = []
        
        for feedback in self.feedback_data:
            if feedback['actual_score'] is not None:
                lead_data = feedback['lead_data'].copy()
                lead_data['target_score'] = feedback['actual_score']
                training_data.append(lead_data)
        
        return pd.DataFrame(training_data)

# Usage example
def main():
    """Example usage of the training system"""
    
    # Initialize training configuration
    config = TrainingConfig(
        test_size=0.2,
        cv_folds=5,
        model_types=['random_forest', 'gradient_boosting', 'linear_regression']
    )
    
    # Create trainer
    trainer = LeadScoringTrainer(config)
    
    # Generate training data
    logger.info("Generating synthetic training data...")
    training_data = trainer.generate_synthetic_training_data(n_samples=5000)
    
    # Train models
    logger.info("Training models...")
    results = trainer.train_models(training_data, optimize_hyperparameters=True)
    
    # Print results
    print("\nModel Performance Results:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  CV Score: {metrics['cv_score_mean']:.4f} ± {metrics['cv_score_std']:.4f}")
        
        print(f"  Top 5 Important Features:")
        for i, (feature, importance) in enumerate(list(metrics['feature_importance'].items())[:5]):
            print(f"    {i+1}. {feature}: {importance:.4f}")
    
    # Test prediction on new lead
    new_lead = {
        'beat_plan_tag': 'NCA_Hot',
        'meeting_days_from_today': 5,
        'purchase_cycle_days_from_today': 45,
        'intent_score': 85,
        'days_since_last_meeting': 3,
        'target_meetings': 4,
        'completed_meetings': 2,
        'annual_turnover': 150,
        'company_size': 'large',
        'industry': 'tech',
        'lead_source': 'linkedin',
        'engagement_level': 'high'
    }
    
    prediction = trainer.predict_lead_score(new_lead, 'random_forest')
    print(f"\nNew Lead Prediction:")
    print(f"Predicted Score: {prediction['predicted_score']:.2f}")
    print(f"Confidence Interval: {prediction['confidence_interval']}")
    
    # Save models
    trainer.save_models("lead_scoring_models")
    
    # Set up real-time learning
    rt_system = RealTimeTrainingSystem(trainer)
    
    # Simulate feedback collection
    rt_system.collect_feedback(
        lead_data=new_lead,
        predicted_score=prediction['predicted_score'],
        actual_score=3.2,  # Simulated actual score
        conversion=True
    )
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()