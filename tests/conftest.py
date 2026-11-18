import os

# Settings requires an API key at import/validation time; tests never call Gemini.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
