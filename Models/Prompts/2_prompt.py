from langchain_core.prompts import ChatPromptTemplate

# Inputs
topic = "Emotional Intelligence"
number_of_lines = 2
style = "Conversational"
language = "English"

# Create prompt
prompt_template = ChatPromptTemplate.from_template(
    "Write a {number_of_lines}-line counselling on the topic of {topic} in {style} style and {language} language."
)

# Format prompt
prompt = prompt_template.format(
    number_of_lines=number_of_lines,
    topic=topic,
    style=style,
    language=language
)

print(prompt)

print(result.content)
