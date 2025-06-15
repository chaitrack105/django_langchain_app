#!/usr/bin/env python3
"""
Working example of lead scoring with database integration
This script demonstrates the complete workflow
"""

from utils import create_lead_prioritization_assistant


def main():
    print("🚀 Lead Scoring System Demo")
    print("=" * 40)
    
    # Step 1: Create assistant with MySQL integration
    print("1. Initializing AI Assistant with MySQL...")
    try:
        assistant = create_lead_prioritization_assistant(use_mysql=True)
        print("✅ Assistant created successfully")
    except Exception as e:
        print(f"❌ Failed to create assistant: {e}")
        return
    
    # Step 2: Train the model
    print("\n2. Training the model...")
    try:
        train_response = assistant.generate_response("train model")
        print(train_response)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 3: Score a specific lead
    print("\n3. Scoring a specific lead...")
    try:
        score_response = assistant.generate_response("score lead abc123")
        print(score_response)
    except Exception as e:
        print(f"❌ Scoring failed: {e}")
        # Try another prospect ID
        try:
            score_response = assistant.generate_response("score lead abc001")
            print(score_response)
        except Exception as e2:
            print(f"❌ Scoring with abc001 also failed: {e2}")
    
    # Step 4: Get top leads
    print("\n4. Getting top leads...")
    try:
        top_leads_response = assistant.generate_response("show me top 5 leads")
        print(top_leads_response)
    except Exception as e:
        print(f"❌ Failed to get top leads: {e}")
    
    # Step 5: Test conversational queries
    print("\n5. Testing conversational queries...")
    test_queries = [
        "Which leads should I prioritize today?",
        "Show me leads with high intent scores",
        "What are the best performing beat tags?",
        "Give me leads from Mumbai"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        try:
            response = assistant.generate_response(query)
            print(f"Assistant: {response[:150]}...")
        except Exception as e:
            print(f"❌ Query failed: {e}")

if __name__ == "__main__":
    main()