import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
                      http_options={"timeout": 20000})   # 20 second limit

try:
    for m in client.models.list():
        print(m.name)
except Exception as e:
    print("ERROR:", str(e)[:400])