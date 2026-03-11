import requests
import json
import sys
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure the backend server is running!")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n" + "="*60)
    print("Testing Root Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Root endpoint working!")
            return True
        else:
            print("❌ Root endpoint failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_predict_endpoint(image_path):
    """Test the prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found at {image_path}")
        return False
    
    try:
        # Open and send the image file
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(f"{API_BASE_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n📊 Prediction Results:")
            print("="*60)
            print(f"Success: {result.get('success')}")
            print(f"Diagnosis: {result.get('diagnosis')}")
            print(f"Confidence: {result.get('confidence')}%")
            print(f"Risk Level: {result.get('risk_level')}")
            
            print("\n📈 Class Probabilities:")
            for class_name, prob in result.get('class_probabilities', {}).items():
                print(f"   {class_name:40} : {prob:6.2f}%")
            
            print(f"\n💡 Recommendation:")
            print(f"   {result.get('recommendation')}")
            
            print(f"\n⚠️  Disclaimer:")
            print(f"   {result.get('disclaimer')}")
            
            print("\n✅ Prediction successful!")
            return True
        else:
            print(f"❌ Prediction failed!")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_invalid_file():
    """Test API response with invalid file"""
    print("\n" + "="*60)
    print("Testing Invalid File Upload")
    print("="*60)
    
    try:
        # Try to upload a text file instead of an image
        files = {'file': ('test.txt', b'This is not an image', 'text/plain')}
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 400:
            print("✅ API correctly rejected invalid file!")
            return True
        else:
            print("⚠️  Expected 400 status code for invalid file")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def run_all_tests(image_path=None):
    """Run all API tests"""
    print("\n" + "="*70)
    print(" "*15 + "CERVICAL CANCER API TEST SUITE")
    print("="*70)
    
    results = {
        "Health Check": test_health_check(),
        "Root Endpoint": test_root_endpoint(),
        "Invalid File": test_invalid_file(),
    }
    
    if image_path:
        results["Prediction"] = test_predict_endpoint(image_path)
    else:
        print("\n⚠️  Skipping prediction test (no image provided)")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} : {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("="*70 + "\n")
    
    return all(results.values())

def interactive_test():
    """Interactive testing mode"""
    print("\n" + "="*70)
    print("INTERACTIVE API TESTING")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("1. Test health check")
        print("2. Test root endpoint")
        print("3. Test prediction with image")
        print("4. Test invalid file upload")
        print("5. Run all tests")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            test_health_check()
        elif choice == '2':
            test_root_endpoint()
        elif choice == '3':
            image_path = input("Enter path to image file: ").strip()
            test_predict_endpoint(image_path)
        elif choice == '4':
            test_invalid_file()
        elif choice == '5':
            image_path = input("Enter path to test image (or press Enter to skip): ").strip()
            run_all_tests(image_path if image_path else None)
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("CERVICAL CANCER DETECTION API - TEST SCRIPT")
    print("="*70)
    
    if len(sys.argv) > 1:
        # Command-line mode with image path
        image_path = sys.argv[1]
        run_all_tests(image_path)
    else:
        # Interactive mode
        print("\nNo image path provided. Starting interactive mode...")
        print("Usage: python test_api.py [path_to_test_image]")
        
        choice = input("\nContinue in interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_test()
        else:
            print("\n💡 Tip: Run quick test with:")
            print("   python test_api.py path/to/your/test_image.jpg")

if __name__ == "__main__":
    main()