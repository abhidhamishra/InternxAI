from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

import streamlit as st
from dotenv import load_dotenv
#import os
load_dotenv()

#HF_TOKEN = os.getenv("HF_TOKEN")

endpoint = HuggingFaceEndpoint( repo_id="Qwen/Qwen2.5-1.5B-Instruct",task="text-generation" )
cm = ChatHuggingFace(llm=endpoint)

st.header("Emotional Intelligence Counselling")


topic = st.text_input("Topic: ", "Emotional Intelligence")
number_of_lines = st.number_input("Number of lines: ", min_value=1, max_value=10, value=2)
style = st.selectbox("Style: ", ["Conversational", "Formal", "Informal"])
language = st.selectbox("Language: ", ["English", "Hindi", "French"])

prompt_template = f"Write a {number_of_lines}-line counselling on the topic of {topic} in {style} style and {language} language."
if st.button("Generate Counselling"):
    result = cm.invoke(prompt_template)
    st.write(result.content)


