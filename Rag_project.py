import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer

# --- Step 1: Load & Process PDF ---
pdf_path = os.path.join(os.path.dirname(__file__), "2411.15594v5.pdf")
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Token-aware splitting with safety buffer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=256,  # Reduced for sequence limits
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
docs = splitter.split_documents(data)
print(f"Number of chunks: {len(docs)}")

# --- Step 2: Embeddings & Vector Store ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(docs, embeddings)

# --- Step 3: LLM Setup with Format Enforcement ---
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=250,  # Increased for full format
    temperature=0.2,
    do_sample=True,      # Required for temperature
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# --- Step 4: Strict Format Prompt ---
prompt_template = """You MUST format your answer as:
Challenge 1: [description]
Challenge 2: [description]

Context: {context}
Question: {question}
Do NOT include any other text in your response."""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- Step 5: QA Chain Configuration ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # More context
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

# --- Step 6: Execute Query ---
query = "What are the two main challenges that hinder the widespread application of the 'LLM-as-a-Judge' approach?"
result = qa_chain.invoke({"query": query})
print("\nFinal Answer:")
print(result["result"])
