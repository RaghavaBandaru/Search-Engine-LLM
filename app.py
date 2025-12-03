import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain import hub

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("🔎 LangChain - Chat with Search (Groq + Llama 3.3 70B)")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.info("Please enter your Groq API key to continue.")
    st.stop()

# -------------------------------------------------------------
# Tools
# -------------------------------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

duckduckgo = DuckDuckGoSearchRun(name="Search")

tools = [duckduckgo, arxiv, wiki]

# -------------------------------------------------------------
# LLM (Groq Llama 3.3-70B)
# -------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",   # UPDATED MODEL
    streaming=True
)

# -------------------------------------------------------------
# ReAct Agent
# -------------------------------------------------------------
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# -------------------------------------------------------------
# Chat History
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can search Arxiv, Wikipedia, and the web using Llama-3.3-70B. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------------------------------------
# Handle user input
# -------------------------------------------------------------
if prompt_text := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.chat_message("user").write(prompt_text)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = agent_executor.invoke(
            {"input": prompt_text},
            callbacks=[cb]
        )

        final_answer = response["output"]
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.write(final_answer)
