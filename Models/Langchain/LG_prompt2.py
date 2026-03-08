from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────
AGENT_NAME = "MindFit AI"
AGENT_AVATAR = "🧠"
STARTER_QUESTIONS = [
    "Give me a brain teaser to solve 🧩",
    "What are the best memory exercises?",
    "Teach me a mindfulness technique 🌿",
    "How does exercise improve brain health?",
    "Give me a cognitive challenge for today",
    "What foods boost brain performance?",
]

SYSTEM_PROMPT = """You are MindFit AI, a friendly and knowledgeable Mental Fitness Coach.
You specialize in brain puzzles, cognitive exercises, mindfulness techniques, and mental wellness.
Keep responses engaging, concise, and actionable. Use emojis to keep the tone warm."""

# ── Model Setup ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation"
    )
    return ChatHuggingFace(llm=endpoint)

cm = load_model()

# ── Helper Functions ─────────────────────────────────────────
def get_quick_puzzle():
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Give me one short, fun brain teaser puzzle (2-3 sentences max). Include the answer hidden at the end with a spoiler label.")
    ]
    response = cm.invoke(messages)
    return f"🧩 **Puzzle of the Day**\n\n{response.content}"

def chat(user_input: str, history: list):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))

    response = cm.invoke(messages)
    updated_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response.content}
    ]
    return response.content, updated_history

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title=f"{AGENT_NAME} — Mental Fitness Coach",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS Styling ───────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f0f4ff 0%, #fdf6ff 100%); }
    .stChatMessage { border-radius: 16px; margin-bottom: 8px; padding: 4px; }
    .main-header { text-align: center; padding: 20px 0 10px 0; }
    .starter-btn {
        background: white; border: 1px solid #d0c4ff;
        border-radius: 20px; padding: 8px 16px; margin: 4px;
        cursor: pointer; font-size: 13px; transition: all 0.2s;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "puzzle_of_day" not in st.session_state:
    st.session_state.puzzle_of_day = get_quick_puzzle()
if "started" not in st.session_state:
    st.session_state.started = False

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindFit AI")
    st.markdown("*Your Personal Mental Fitness Coach*")
    st.divider()
    st.markdown("### 🎯 What I Can Help With")
    st.markdown("""
    - 🧩 **Brain Puzzles & Teasers**
    - 🏋️ **Cognitive Exercises**
    - 🌿 **Mindfulness Techniques**
    - 📊 **Benefits & Science**
    - 🎓 **Learning Strategies**
    - 👴 **Age-specific Brain Health**
    """)
    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Messages Exchanged", len(st.session_state.display_messages))
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.display_messages = []
        st.session_state.started = False
        st.rerun()
    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.caption("MindFit AI provides general wellness information only. For clinical mental health concerns, please consult a qualified professional.")

# ── Main Header ──────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🧠 MindFit AI</h1>
    <p style='color: #6b7280; font-size: 16px;'>Your Personal Mental Fitness & Brain Health Coach</p>
</div>
""", unsafe_allow_html=True)

# ── Welcome Screen ────────────────────────────────────────────
if not st.session_state.started:
    st.info(st.session_state.puzzle_of_day)
    st.markdown("### 💬 Try asking me about...")
    cols = st.columns(2)
    for i, question in enumerate(STARTER_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, use_container_width=True, key=f"starter_{i}"):
                st.session_state.started = True
                st.session_state.display_messages.append({"role": "user", "content": question})
                with st.spinner("🧠 Thinking..."):
                    reply, updated_history = chat(question, st.session_state.conversation_history)
                st.session_state.conversation_history = updated_history
                st.session_state.display_messages.append({"role": "assistant", "content": reply})
                st.rerun()
    st.markdown("---")

# ── Chat Display ─────────────────────────────────────────────
for message in st.session_state.display_messages:
    avatar = "👤" if message["role"] == "user" else AGENT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ── Chat Input ───────────────────────────────────────────────
user_input = st.chat_input(placeholder="Ask me about puzzles, brain exercises, mindfulness, or mental fitness benefits...")

if user_input:
    st.session_state.started = True
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        with st.spinner("🧠 Thinking..."):
            reply, updated_history = chat(user_input, st.session_state.conversation_history)
        st.markdown(reply)
    st.session_state.conversation_history = updated_history
    st.session_state.display_messages.append({"role": "assistant", "content": reply})