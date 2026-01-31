from fraud_service import load_voice_model

print("Testing model download/load...")
try:
    feat, model, device = load_voice_model()
    if model:
        print("SUCCESS: Model loaded.")
    else:
        print("FAILURE: Model returned None.")
except Exception as e:
    print(f"CRASH: {e}")
