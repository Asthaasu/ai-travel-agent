import streamlit as st
from agents.travel_agent import run_agent
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Agent",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
    }
    .assistant-message {
        background-color: #f0f9f0;
        border-left: 4px solid #27ae60;
    }
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>✈️ AI Travel Agent</h1>
    <p>Powered by LangGraph & GPT-4o-mini | Plan your dream trip instantly!</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774278.png", width=100)
    st.title("🌍 Travel Assistant")
    
    st.markdown("### 📧 Send Plan to Email")
    user_email = st.text_input(
        "Your Email (optional)",
        placeholder="you@example.com",
        help="Enter your email to receive the travel plan"
    )
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    
    examples = [
        "Plan a 7-day trip to Paris from Mumbai in December",
        "I want to travel to Tokyo from Delhi in March for 10 days",
        "Weekend trip to Goa from Pune next month",
        "Family trip to Singapore for 5 days in January"
    ]
    
    for example in examples:
        if st.button(f"📍 {example[:40]}...", key=example):
            st.session_state.example_query = example
    
    st.markdown("---")
    st.markdown("### 🛠️ Powered By")
    st.markdown("""
    - 🤖 GPT-4o-mini (OpenAI)
    - 🕸️ LangGraph
    - 🔍 SERPAPI
    - 📧 SendGrid
    - 🎨 Streamlit
    """)
    
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle example query click
if "example_query" in st.session_state:
    example = st.session_state.pop("example_query")
    st.session_state.pending_input = example

# ─────────────────────────────────────────
# CHAT DISPLAY
# ─────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #888;">
            <h3>👋 Welcome! I'm your AI Travel Agent</h3>
            <p>Tell me where you want to go and I'll plan everything for you!</p>
            <p><b>Try:</b> "I want to travel to Amsterdam from Delhi from Dec 20-28"</p>
        </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="✈️"):
                st.markdown(message["content"])

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Where would you like to travel? ✈️") or pending

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    
    # Get agent response
    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("🔍 Searching flights, hotels & attractions..."):
            try:
                response, updated_history = run_agent(
                    user_input,
                    st.session_state.chat_history,
                    user_email
                )
                st.session_state.chat_history = updated_history
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nPlease check your API keys in the .env file."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
