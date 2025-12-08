import streamlit as st
from langchain_classic.memory import ConversationBufferMemory
from main import (
    populate_kb,
    build_vs,
    qa_chain,
    web_search_tool,
    web_qa_chain,
    FAISS,
    HuggingFaceEmbeddings
)

# Streamlit Setup
st.set_page_config(
    page_title="Conversational Knowledge Bot",
    layout="centered",
)

st.title("Conversational Knowledge Bot")

st.write("""
Choose how you want the bot to answer your questions.
""")

#============ MODE SELEcTION ============
mode = st.radio(
    "Select Mode:",
    ["Static KB Mode", "Web Search Mode"]
)

st.divider()

# initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Initialize chat history UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    

#=========== STATIC KB MODE ==============
if mode == "Static KB Mode":
    st.subheader("Static Knowledge Base Mode")

    @st.cache_resource(show_spinner=True)
    def load_vectorstore():
        try:
            vs = FAISS.load_local(
                "./data",
                embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                allow_dangerous_deserialization=True
            )
            return vs
        except:
            kb = populate_kb()
            vs = build_vs(kb)
            return vs

    vectorstore = load_vectorstore()

    # Build Conversational retrieval Chain
    chain = qa_chain(vectorstore, memory=st.session_state.memory)
    query = st.chat_input("Ask a question about Indian Nobel laureates:")

    if query:
        with st.spinner("Thinking..."):
            answer = chain({"question": query})["answer"]
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))
        

#=========== WEB SEARCH MODE =============
else:
    st.subheader("Web Search Mode")

    # Load DuckDuckGo tool
    search_tool = web_search_tool()
    query = st.chat_input("Ask anything (web search enabled):")

    if query:
        with st.spinner("Searching & thinking..."):
            answer = web_qa_chain(
                question=query,
                search_tool=search_tool,
                memory=st.session_state.memory
            )
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))
        
# Display Chat
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
