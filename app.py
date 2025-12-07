import streamlit as st
from main import (
    populate_kb,
    build_vs,
    qa_chain,
    ask,
    ChatOpenAI,
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
Ask anything about Indian Nobel laureates!
""")

# Load Vectorstore
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    try:
        vs = FAISS.load_local(
            "/data",
            embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
        return vs
    except:
        # first time: populate and build
        kb = populate_kb()
        vs = build_vs(kb)
        return vs

vectorstore = load_vectorstore()

# Create QA Chain with Conversational Memory
chain = qa_chain(vectorstore)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        answer = chain({"question": query})["answer"]

    # update chat UI memory
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Bot", answer))

# Display Chat
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

