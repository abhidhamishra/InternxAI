from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    
)

chat = ChatHuggingFace(llm=endpoint)

# Inputs
topic = input("Topic: ")
number_of_lines = input("Number of lines: ")
style = input("Style: Conversational/Formal")
language = input("language: ")

# Prompt
prompt = f"""
Write a {number_of_lines}-line counselling on the topic of {topic}
in {style} style and in {language} language.
"""

# Single invoke only
result = chat.invoke(prompt)

print(result.content)