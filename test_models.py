#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""

import os
import sys

def test_model_loading():
    """Test if models can be loaded"""
    print("🧪 Testing model loading...")
    
    # Check if model files exist
    required_files = ["firensmoke.pt", "PPEdetect.pt"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            return False
        else:
            print(f"✅ Found: {file}")
    
    # Try importing and loading models
    try:
        print("\n🔄 Testing model loader...")
        from model_loader import load_yolo_model
        
        print("🔄 Loading fire/smoke model...")
        fire_model = load_yolo_model("firensmoke.pt")
        print("✅ Fire/smoke model loaded!")
        
        print("🔄 Loading PPE model...")
        ppe_model = load_yolo_model("PPEdetect.pt")
        print("✅ PPE model loaded!")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_dependencies():
    """Test if all dependencies are available"""
    print("🧪 Testing dependencies...")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("ultralytics", "YOLO"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
        ("pygame", "Pygame"),
        ("torch", "PyTorch")
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            all_good = False
    
    return all_good

def main():
    print("🚀 Fire & PPE Detection - Model Test")
    print("=" * 40)
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Some dependencies are missing!")
        print("Run: python setup.py")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading failed!")
        return False
    
    print("\n🎉 All tests passed!")
    print("✅ Ready to run: python main2.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 