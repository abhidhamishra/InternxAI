from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import re

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    max_new_tokens=1200
)
cm = ChatHuggingFace(llm=endpoint)

st.header("Mental Health Chat Agent | Just for your Mental Fitness 🧠")

topic = st.text_input("Topic: ", "Mental Fitness")
focus_area = st.multiselect("Focus Area: ",
    ["Stress Management", "Anxiety Relief", "Memory Enhancement",
     "Emotional Regulation", "Sleep Improvement", "Confidence Building"],
    default=["Stress Management"])
number_of_lines = st.number_input("Response lines per step: ", min_value=1, max_value=3, value=1)
style = st.selectbox("Style: ", ["Conversational", "Formal"])
language = st.selectbox("Language: ", ["English", "Hindi", "Marathi", "Spanish", "French"])
age_group = st.selectbox("Age Group: ", ["Children (6-12)", "Teenagers (13-19)", "Adults (20-40)", "Middle-aged (41-60)", "Seniors (60+)"])
session_duration = st.slider("Session Duration (minutes): ", min_value=5, max_value=60, value=15, step=5)
difficulty_level = st.select_slider("Difficulty Level: ", options=["Beginner", "Intermediate", "Advanced", "Expert"])
mood_raw = st.radio("Current Mood: ", ["😊 Happy", "😟 Anxious", "😔 Sad", "😠 Frustrated", "😐 Neutral"], horizontal=True)


mood = re.sub(r'[^\w\s]', '', mood_raw).strip()

template = ChatPromptTemplate.from_messages([
    ("system", """You are a professional mental health and cognitive wellness expert.
Your role is to provide clear, detailed, actionable, and compassionate guidance.

STRICT RULES:
- NEVER ask questions in your response.
- ALWAYS provide information in clearly numbered steps.
- Each step must be a direct instruction or explanation — never a question.
- Use warm, supportive, and encouraging language.
- Tailor all content to the user's age group, mood, difficulty level, and focus area.
- Respond entirely in {language}.
- Writing style: {style}.
- Each step must be exactly {number_of_lines} sentence(s) long. Example of 1-line step: "Sit comfortably and close your eyes."
- Total content should be completable within {session_duration} minutes.
"""),
    ("human", """Provide a step-by-step mental wellness guide on: **{topic}**

User Profile:
- Age Group: {age_group}
- Current Mood: {mood}
- Focus Areas: {focus_area}
- Difficulty Level: {difficulty_level}
- Session Duration: {session_duration} minutes

Write exactly these 6 sections. Under each, give numbered steps only — no questions.

1. 🧠 Understanding {topic}
   - What it is and why it matters for mental fitness.

2. 🏋️ Practical Exercises
   - Exercises: {Exercises}
   - Step-by-step instructions for each.

3. 🎮 Mind Games & Puzzles
   - Games: {Mind_Games} | Puzzles: {Puzzles}
   - Clear how-to steps for each activity.

4. 🔬 Science Behind It
   - {Science}
   - Simple, digestible scientific explanations.

5. ✅ Why It's Necessary
   - {Necessity}
   - Reasons tailored to age group: {age_group} and mood: {mood}.

6. 📅 {session_duration}-Minute Daily Action Plan
   - A timed routine broken into steps (e.g., "0–3 min: ...", "3–7 min: ...").

Do NOT ask any questions. Only provide steps and information.
""")
])

if st.button("Mental Health Assistance"):
    chain = template | cm

    invoke_params = {
        "topic": topic,
        "Mind_Games": "Memory games, logic puzzles, pattern recognition",
        "Exercises": "Mindfulness meditation, cognitive training (attention & focus), dual-task neurobic activities, breathing techniques",
        "Puzzles": "Sudoku, crosswords, riddles, lateral thinking challenges",
        "Science": "Neuroplasticity basics, cognitive decline prevention, mental fitness across life stages, key research findings",
        "Necessity": "Mental fitness supports well-being, emotional resilience, cognitive function, and quality of life at all ages.",
        "number_of_lines": str(number_of_lines),
        "style": style,
        "language": language,
        "age_group": age_group,
        "session_duration": str(session_duration),
        "difficulty_level": difficulty_level,
        "focus_area": ", ".join(focus_area),
        "mood": mood,  
    }

    with st.spinner("Generating your personalized mental wellness guide..."):
        result = chain.invoke(invoke_params)

    st.success("Your Mental Wellness Guide is Ready!")
    st.markdown("---")
    st.markdown(result.content)
    st.markdown("---")
    st.caption("💡 Tip: Practice these steps consistently for best results. Your mental fitness journey starts today!")