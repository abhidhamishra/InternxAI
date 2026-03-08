from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chatmodel=ChatAnthropic(model='claude-2')

result = chatmodel.invoke('Where is Lucknow located?')

print(result)
