from langchain_google_genai   import ChatGoogleGenerativeAI
from dotenv import load_dotenv  

load_dotenv()

chat = ChatGoogleGenerativeAI(model="gemini-3-flash")

result = chat.invoke('Where is Lucknow located?')

print(result)






