import requests
import os

BASE_URL = "http://127.0.0.1:8000"

def test_files_list():
    print("\n--- Testing GET /files ---")
    res = requests.get(f"{BASE_URL}/files")
    print("Status Code:", res.status_code)
    print("Response:", res.json())

def test_upload():
    print("\n--- Testing POST /upload ---")
    files = [
        ("files", ("johnson_v_smith.pdf", open("test_cases/johnson_v_smith.pdf", "rb"), "application/pdf")),
        ("files", ("davis_v_mega_corp.pdf", open("test_cases/davis_v_mega_corp.pdf", "rb"), "application/pdf")),
    ]
    res = requests.post(f"{BASE_URL}/upload", files=files)
    print("Status Code:", res.status_code)
    print("Response:", res.json())

def test_query_low_confidence():
    print("\n--- Testing Unrelated Query (Low Confidence) ---")
    payload = {"question": "What is the speed of light in a vacuum?"}
    res = requests.post(f"{BASE_URL}/query", json=payload)
    print("Status Code:", res.status_code)
    print("Response JSON:")
    import pprint
    pprint.pprint(res.json())

def test_query_high_confidence():
    print("\n--- Testing Related Query (High Confidence - Should trigger OpenAI auth error) ---")
    payload = {"question": "What was the verdict in the breach of contract claim against Alice Smith?"}
    res = requests.post(f"{BASE_URL}/query", json=payload)
    print("Status Code:", res.status_code)
    print("Response JSON:")
    import pprint
    pprint.pprint(res.json())

if __name__ == "__main__":
    # Set a dummy key in the environment before running tests
    os.environ["GEMINI_API_KEY"] = "mock-gemini-key-12345"
    
    test_files_list()
    
    # Try querying with unrelated text
    try:
        test_query_low_confidence()
    except Exception as e:
        print("Low confidence query failed:", e)
        
    # Try querying with related text
    try:
        test_query_high_confidence()
    except Exception as e:
        print("High confidence query failed:", e)

