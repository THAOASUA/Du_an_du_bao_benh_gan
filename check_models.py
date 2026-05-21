import os
import joblib

print("=== CHECKING MODELS ===\n")

# Kiểm tra file trong thư mục hiện tại
print(f"Current directory: {os.getcwd()}")
print("\nAll .pkl files:")
for file in os.listdir('.'):
    if file.endswith('.pkl'):
        size = os.path.getsize(file)
        print(f"  - {file} ({size} bytes)")

# Thử load model
print("\n=== TRYING TO LOAD MODELS ===")

try:
    if os.path.exists("liver_model.pkl"):
        lgb = joblib.load("liver_model.pkl")
        print("✅ liver_model.pkl loaded successfully")
    else:
        print("❌ liver_model.pkl not found")
except Exception as e:
    print(f"❌ Error loading liver_model.pkl: {e}")

try:
    if os.path.exists("rf_model.pkl"):
        rf = joblib.load("rf_model.pkl")
        print("✅ rf_model.pkl loaded successfully")
    else:
        print("❌ rf_model.pkl not found")
except Exception as e:
    print(f"❌ Error loading rf_model.pkl: {e}")