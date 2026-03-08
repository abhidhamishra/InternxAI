from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv  
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
result = embeddings.embed_query("Where is Eiffel Tower located?")
print(result)
