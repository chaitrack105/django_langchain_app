# simple_test.py
import sys
import os
sys.path.append('.')

from simplified_mysql_integration import DatabaseConfig, SimplifiedMySQLLeadManager

# Test database connection
print("Testing database connection...")
db_config = DatabaseConfig.from_env()
print(f"Connecting to: {db_config.host}:{db_config.port}/{db_config.database}")

try:
    db_manager = SimplifiedMySQLLeadManager(db_config)
    print("✅ Database connected!")
    
    # Fetch sample data
    data = db_manager.fetch_training_data(limit=5)
    print(f"✅ Fetched {len(data)} records")
    print("Columns:", list(data.columns))
    
except Exception as e:
    print(f"❌ Error: {e}")