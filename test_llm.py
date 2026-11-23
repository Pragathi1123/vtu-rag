import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
                      http_options={"timeout": 20000})
model = os.getenv("GEMINI_MODEL")
print("Using model:", model)

start = time.time()
try:
    r = client.models.generate_content(model=model, contents="Say hello in one short sentence.")
    print("Reply:", r.text)
except Exception as e:
    print("ERROR:", str(e)[:400])
print("Took", round(time.time() - start, 1), "seconds")