import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langgraph.prebuilt import create_react_agent

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("🔎 Search Chatbot (Groq + Llama 3.3 70B + LangGraph)")

st.sidebar.title("Settings")
groq_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_key:
    st.info("Please enter your Groq API Key")
    st.stop()

# -------------------------------------------------------------
# Tools
# -------------------------------------------------------------
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1))
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
duck_tool = DuckDuckGoSearchRun(name="Search")

tools = [duck_tool, arxiv_tool, wiki_tool]

# -------------------------------------------------------------
# LLM (Groq Llama 3.3-70B)
# -------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.3-70b-versatile",
    streaming=True
)

# -------------------------------------------------------------
# LangGraph ReAct Agent
# -------------------------------------------------------------
agent = create_react_agent(llm=llm, tools=tools)

# -------------------------------------------------------------
# Chat History
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I can search Arxiv, Wikipedia, and the web using Llama-3.3-70B. Ask me anything!"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------------------------------------
# Handle Input
# -------------------------------------------------------------
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        result = agent.invoke(
            {"input": user_input},
            callbacks=[cb]
        )

        answer = result["output"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
