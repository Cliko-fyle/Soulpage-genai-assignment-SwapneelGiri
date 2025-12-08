import streamlit as st
from langchain_classic.memory import ConversationBufferMemory
from main import web_search_tool, web_qa_chain

# Streamlit Setup
st.set_page_config(
    page_title="Conversational Knowledge Bot",
    layout="centered",
)

st.title("Conversational Knowledge Bot")

st.write("""Ask anything!""")


@st.cache_resource
def search_tool():
    return web_search_tool()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key = "question",
        output_key="output")

# Initialize chat history for display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history FIRST
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

query = st.chat_input("Ask anything (web search enabled):")

if query:
    # Display user message immediately
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching & thinking..."):
            answer = web_qa_chain(
                question=query,
                search_tool=search_tool(),
                memory=st.session_state.memory
            )
        st.write(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
    

# #============ MODE SELEcTION ============
# mode = st.radio(
#     "Select Mode:",
#     ["Static KB Mode", "Web Search Mode"],
#     index = None
# )

# st.divider()
    

# #=========== STATIC KB MODE ==============
# if mode == "Static KB Mode":
#     st.subheader("Static Knowledge Base Mode")
#     # initialize memory
#     if "memory" not in st.session_state:
#         st.session_state.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             output_key="answer")

#     @st.cache_resource(show_spinner=True)
#     def load_vectorstore():
#         try:
#             vs = FAISS.load_local(
#                 "./data",
#                 embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#                 allow_dangerous_deserialization=True
#             )
#             return vs
#         except:
#             kb = populate_kb()
#             vs = build_vs(kb)
#             return vs

#     vectorstore = load_vectorstore()

#     # Build Conversational retrieval Chain
#     chain = qa_chain(vectorstore, memory=st.session_state.memory)
#     query = st.chat_input("Ask a question about Indian Nobel laureates:")

#     if query:
#         with st.spinner("Thinking..."):
#             answer = chain({"question": query})["answer"]
#         st.session_state.chat_history.append(("You", query))
#         st.session_state.chat_history.append(("Bot", answer))

