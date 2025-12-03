import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import create_agent   # correct import


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("🔎 Search Chatbot (Groq + Llama-3.3-70B + LangGraph)")
st.sidebar.title("Settings")

groq_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_key:
    st.info("Please enter your Groq API Key to continue")
    st.stop()


# -------------------------------------------------------------
# Tools
# -------------------------------------------------------------
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1))
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
duck_tool = DuckDuckGoSearchRun(name="Search")

tools = [duck_tool, arxiv_tool, wiki_tool]


# -------------------------------------------------------------
# LLM (Groq)
# -------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=2048,
    streaming=True,
)


# -------------------------------------------------------------
# Create LangGraph Agent
# -------------------------------------------------------------
agent = create_agent(
    model=llm,
    tools=tools,
)


# -------------------------------------------------------------
# Streamlit Chat UI (session memory)
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can search ArXiv, Wikipedia, and the Web for you."}
    ]

# Render chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# -------------------------------------------------------------
# Chat Input
# -------------------------------------------------------------
user_input = st.chat_input("Ask me anything…")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Agent input state
        state = {
            "messages": [{"role": "user", "content": user_input}],
            "input": user_input
        }

        result = agent.invoke(state, callbacks=[cb])

        # ---------------------------------------------------------
        # UNIVERSAL SAFE ANSWER EXTRACTION (fixes all errors)
        # ---------------------------------------------------------
        if hasattr(result, "content"):
            answer = result.content

        elif isinstance(result, dict):
            if "output" in result:
                answer = result["output"]
            elif "messages" in result and len(result["messages"]) > 0:
                last = result["messages"][-1]
                if isinstance(last, dict):
                    answer = last.get("content", "No content.")
                else:
                    answer = getattr(last, "content", "No content.")
            else:
                answer = "No valid response returned."
        else:
            answer = str(result)

        # Save to session + display
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
