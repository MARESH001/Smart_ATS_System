import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_KEY"))

# Fetch available models
response = genai.list_models()  # This is a generator

# Convert the generator to a list and print the available models
models_list = list(response)
print(models_list)