try:
    from streamlit_audiorecorder import audiorecorder
    print("SUCCESS: Import worked")
except Exception as e:
    print(f"FAILURE: {e}")
