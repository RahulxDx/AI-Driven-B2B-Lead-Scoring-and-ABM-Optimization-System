"""
Test script to check API health and diagnose issues
"""

import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing API endpoints...")
    print("=" * 60)
    
    # Test health endpoint
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"[OK] Health check passed:")
            print(f"   Status: {health.get('status')}")
            print(f"   Context Builder Ready: {health.get('context_builder_ready')}")
            print(f"   Query Engine Ready: {health.get('query_engine_ready')}")
        else:
            print(f"[ERROR] Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to API. Is the server running?")
        print("   Start it with: python start_api.py")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False
    
    # Test chat endpoint
    print("\n2. Testing /chat endpoint...")
    try:
        test_query = "What are the top industries?"
        response = requests.post(
            f"{base_url}/chat",
            json={"query": test_query, "top_k": 5},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Chat endpoint working!")
            print(f"   Response: {result.get('response', '')[:100]}...")
            print(f"   Contexts used: {result.get('contexts_used', 0)}")
        elif response.status_code == 500:
            try:
                error = response.json()
                print(f"[ERROR] API Error 500:")
                print(f"   Detail: {error.get('detail', 'Unknown error')}")
            except:
                print(f"[ERROR] API Error 500: {response.text}")
        else:
            print(f"[ERROR] Chat endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"[ERROR] Error testing chat: {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Testing complete!")
    
    return True

if __name__ == "__main__":
    test_api()

