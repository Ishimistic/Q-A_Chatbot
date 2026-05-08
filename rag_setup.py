from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import os

# Disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ---- 1. Load and split documents ----
loader = WebBaseLoader(web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# ---- 2. Create embeddings & vectorstore ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# ---- 3. HuggingFace LLM pipeline ----
hf_model = pipeline(
      "text2text-generation",
    model="google/flan-t5-base",  # Larger model for better answers
    device=-1,  # CPU (-1) or GPU (0)
    max_new_tokens=800,           # Allow longer responses
    temperature=0.7      
)
retriever.search_kwargs["k"] = 5
llm = HuggingFacePipeline(pipeline=hf_model)

# ---- 4. Custom Prompt (no LangSmith) ----
prompt_template = """
Use the following context to answer the question.
If you don't know the answer, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ---- 5. Helper to format docs ----
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# ---- 6. Build RAG chain ----
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- 7. Function to query ----
def ask_question(question: str) -> str:
    return rag_chain.invoke(question)
