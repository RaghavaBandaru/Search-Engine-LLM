import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import initialize_agent, AgentType


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("🔎 Search Chatbot (Groq + Llama-3.3-70B)")
st.sidebar.title("Settings")

groq_key = st.sidebar.text_input("Groq API Key:", type="password")
if not groq_key:
    st.stop()


# -------------------------------------------------------------
# LLM
# -------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.3-70b-versatile",
    streaming=True,
    temperature=0.2,
    max_tokens=2048,
)


# -------------------------------------------------------------
# Tools (text-only ReAct tools)
# -------------------------------------------------------------
tools = [
    DuckDuckGoSearchRun(name="search"),
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1)),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1)),
]


# -------------------------------------------------------------
# Classic ReAct Agent (Works perfectly with Groq)
# -------------------------------------------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)


# -------------------------------------------------------------
# Streamlit Chat
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything…"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


user_input = st.chat_input("Ask me anything…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = agent.run(user_input, callbacks=[cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
