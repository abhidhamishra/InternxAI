from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-3-flash")

result = embeddings.embed_query("Where is Eiffel Tower located?")   
print(result)
