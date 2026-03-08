from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  

load_dotenv()   

chat = ChatOpenAI(model='gpt-3.5-turbo-instruct')
result = chat.invoke('Where is Lucknow located?')
print(result)
