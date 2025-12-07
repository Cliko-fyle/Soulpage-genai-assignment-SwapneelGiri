import os, pathlib, json
import wikipediaapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st

# function to build knowledge base
def populate_kb():

  laureates = [
      "Rabindranath Tagore",
      "C. V. Raman",
      "Har Gobind Khorana",
      "Mother Teresa",
      "Amartya Sen",
      "Kailash Satyarthi",
      "Venkatraman Ramakrishnan",
      "Abhijit Banerjee",
      "Subrahmanyan Chandrasekhar",
      "Ronald Ross"
      ] # 10 indian nobel laureates
  kb = []

  wiki = wikipediaapi.Wikipedia(user_agent = "swapneelgiri5@gmail.com", language = "en")

  for name in laureates:
    page = wiki.page(name)

    if not page.exists():
      print(f"page not found for {name}")
      continue
    entry = {
          "name": name,
          "content": page.text,
          "url": page.fullurl
      }
    kb.append(entry)
    print(f"Added: {name} → {page.title}")
  return kb

# function to build vectorstore
def build_vs(kb):
  embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  splitter = RecursiveCharacterTextSplitter(
      chunk_size = 400,
      chunk_overlap = 50
  )

  docs = []
  for entry in kb:
    pieces = splitter.split_text(entry["content"])
    for chunk in pieces:
      docs.append({
          "text": chunk,
          "metadata": {"name": entry["name"]}
          })

  texts = [d["text"] for d in docs]
  metadatas = [d["metadata"] for d in docs]

  vectorstore = FAISS.from_texts(
      texts = texts,
      embedding = embedding_model,
      metadatas = metadatas)
  vectorstore.save_local("/data")

  return vectorstore

# Conversational Bot with Memory
groq_api_key = st.secrets["groq_api_key"]

# Custom LLM
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    temperature = 0.0,
    top_p = 0.7,
    max_completion_tokens = 100)

prompt_template = """
You are a factual question-answer assistant specializing in Indian Nobel laureates.
Use ONLY the retrieved context. Never guess or hallucinate.

If the user uses pronouns like "he", "she", or "they", assume they refer to the
person mentioned earlier in the conversation.

If the context doesn't contain the answer, say "I don't know".

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:
"""
qa_prompt = PromptTemplate(
    input_variables = ["context", "chat_history", "question"],
    template = prompt_template,
)

# function to build Q&A chain
def qa_chain(vectorstore, memory = None):
  # conversational memory
  if memory is None:
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True,
        output_key="answer"
        )

  retriever = vectorstore.as_retriever(search_kwargs = {"k": 5})

  retriever_chain = ConversationalRetrievalChain.from_llm(
      llm = llm,
      retriever = retriever,
      memory = memory,
      combine_docs_chain_kwargs={"prompt": qa_prompt},
      return_source_documents=False
      )
  return retriever_chain

# function to ask question
def ask(question):
  ans = chain({"question": question})
  return ans["answer"]

# Run Q&A bot
vectorestore = build_vs(populate_kb())

vectorstore = FAISS.load_local(
    "/data",
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

chain = qa_chain(vectorstore)