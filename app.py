import os

import streamlit as st
from groq import Groq
from groq import APIConnectionError, APIStatusError, RateLimitError

# ---------------------------------------------------------
# SAAD KHAN - AI Study Assistant
# ---------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are SAAD KHAN, a friendly, patient, and knowledgeable AI study assistant.
Your job is to help students understand concepts and learn effectively.

Guidelines:
- Explain difficult topics in simple, student-friendly language.
- Break complex ideas into clear steps.
- Use examples, analogies, and short summaries when useful.
- Help with mathematics and science step by step, showing the working.
- When asked to solve homework, explain the method instead of only giving an answer.
- Create quizzes, flashcards, revision notes, and practice questions when requested.
- If a question is unclear, ask a short clarifying question.
- Be honest when you are uncertain; never invent facts or sources.
- Adapt the depth of your explanation to the student's level.
- Keep answers organized with headings and bullet points where appropriate.
"""

st.set_page_config(
    page_title="SAAD KHAN | AI Study Assistant",
    page_icon="📚",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #9aa4b2;
            margin-top: 0.25rem;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📚 SAAD KHAN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Your personal AI study assistant</div>',
    unsafe_allow_html=True,
)

# Read the API key from Streamlit Secrets first, then environment variables.
try:
    api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key = ""

api_key = api_key or os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Study assistant")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.markdown(
        """
        **I can help you with:**
        - Understanding difficult topics
        - Mathematics and science problems
        - Notes and summaries
        - Quizzes and flashcards
        - Exam preparation
        """
    )
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by Groq")

if not api_key:
    st.error("Groq API key not found.")
    st.markdown(
        """
        Add your key to Streamlit:

        **For local development**, create `.streamlit/secrets.toml`:
        ```toml
        GROQ_API_KEY = "your_groq_api_key_here"
        ```

        **For Streamlit Community Cloud**, open your app's
        **Settings → Secrets** and add:
        ```toml
        GROQ_API_KEY = "your_groq_api_key_here"
        ```

        Do not upload your API key to GitHub or put it directly in `app.py`.
        """
    )
    st.stop()

client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome screen appears until the first message is sent.
if not st.session_state.messages:
    st.info(
        "Hello! I'm SAAD KHAN, your AI study assistant. "
        "Ask me a question, share a topic, or tell me what you are studying."
    )
    st.markdown("**Try asking:**")
    examples = [
        "Explain photosynthesis in simple words.",
        "Teach me quadratic equations step by step.",
        "Make 5 multiple-choice questions about the solar system.",
    ]
    cols = st.columns(3)
    for col, prompt in zip(cols, examples):
        with col:
            if st.button(prompt, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Support both the chat input and the example prompt buttons.
prompt = st.chat_input("Ask SAAD KHAN anything about your studies...")
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            api_messages.extend(st.session_state.messages)

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.6,
                max_completion_tokens=4096,
                stream=True,
            )

            answer_parts = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    answer_parts.append(chunk.choices[0].delta.content)
                    response_placeholder.markdown("".join(answer_parts) + "▌")

            answer = "".join(answer_parts).strip()
            if not answer:
                answer = "I couldn't generate a response. Please try asking again."

            response_placeholder.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except RateLimitError:
            error_message = (
                "Groq rate limit reached. Please wait a little and try again."
            )
            response_placeholder.error(error_message)
        except APIConnectionError:
            error_message = (
                "Could not connect to Groq. Check your internet connection and try again."
            )
            response_placeholder.error(error_message)
        except APIStatusError as exc:
            error_message = f"Groq returned an API error (status {exc.status_code})."
            response_placeholder.error(error_message)
        except Exception as exc:
            # Show a safe, useful error without exposing the API key.
            response_placeholder.error(
                f"Something went wrong: {type(exc).__name__}. Please try again."
            )
