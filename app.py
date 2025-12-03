import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import create_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("🔎 Search Chatbot (Groq + Llama-3.3-70B + LangGraph)")

groq_key = st.sidebar.text_input("Groq API Key:", type="password")
if not groq_key:
    st.stop()


# -------------------------------------------------------------
# Tools
# -------------------------------------------------------------
tools = [
    DuckDuckGoSearchRun(name="Search"),
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1)),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1)),
]


# -------------------------------------------------------------
# LLM (Groq) + Safety Wrapper
# -------------------------------------------------------------
base_llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=2048,
    streaming=True,
)

# ---- FIX: Ensure messages ALWAYS contains content ----
def ensure_valid_messages(input_messages):
    if not input_messages or len(input_messages) == 0:
        return [{"role": "system", "content": "You are a helpful AI assistant."}]

    # If last message is empty → patch it
    last = input_messages[-1]
    if "content" not in last or not last["content"]:
        last["content"] = "Hello, please continue."

    return input_messages


safe_llm = RunnableLambda(lambda messages: base_llm.invoke(ensure_valid_messages(messages)))


# -------------------------------------------------------------
# Agent
# -------------------------------------------------------------
agent = create_agent(
    model=safe_llm,   # <<-- the important fix
    tools=tools,
)


# -------------------------------------------------------------
# Streamlit Chat UI
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything…"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# -------------------------------------------------------------
# Handle Input
# -------------------------------------------------------------
user_input = st.chat_input("Ask me anything…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # LangGraph input
        state = {
            "messages": [{"role": "user", "content": user_input}],
            "input": user_input
        }

        result = agent.invoke(state, callbacks=[cb])

        # Extract content safely
        if hasattr(result, "content"):
            answer = result.content
        else:
            answer = str(result)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
