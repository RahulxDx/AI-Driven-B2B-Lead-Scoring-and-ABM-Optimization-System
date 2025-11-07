"""
Fix script for sentence-transformers and huggingface_hub compatibility issue
Run this script to fix the import error
"""

import subprocess
import sys

def fix_imports():
    """Fix the sentence-transformers and huggingface_hub compatibility issue"""
    print("🔧 Fixing sentence-transformers and huggingface_hub compatibility...")
    print("=" * 60)
    
    try:
        # Uninstall conflicting packages
        print("\n1. Uninstalling conflicting packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "sentence-transformers", "huggingface-hub"])
        
        # Install compatible versions
        print("\n2. Installing compatible versions...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "huggingface-hub>=0.16.0,<0.20.0",
            "sentence-transformers>=2.2.0,<3.0.0"
        ])
        
        print("\n✅ Successfully fixed compatibility issue!")
        print("\nYou can now run the application:")
        print("  python start_api.py")
        print("  streamlit run app.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during installation: {e}")
        print("\nTry running manually:")
        print("  pip uninstall -y sentence-transformers huggingface-hub")
        print("  pip install 'huggingface-hub>=0.16.0,<0.20.0' 'sentence-transformers>=2.2.0,<3.0.0'")
        return False
    
    return True

if __name__ == "__main__":
    fix_imports()

