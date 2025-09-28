import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()
print("loading...", flush=True)
# Step 1: Setup Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change to any supported Groq model

print("STEP 1: Setting up LLM...", flush=True)
llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)


# Step 2: Connect LLM with FAISS and Create chain

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Build RAG chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# Document combiner chain (stuff documents into prompt)
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# Retrieval chain (retriever + doc combiner)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

print("✅ RAG chain ready", flush=True)

# Now invoke with a single query
while True:
    user_query = input("You: ")

    # exit condition
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! 👋")
        break

    # get response from RAG chain
    response = rag_chain.invoke({"input": user_query})

    # check for valid answer
    if isinstance(response, dict) and "answer" in response:
        answer = response["answer"].strip()
        if answer and "I don't know" not in answer:
            print("Chatbot:", answer)
        else:
            print("Chatbot: Sorry, your PDF data does not have information about this query.")
    else:
        print("Chatbot: Sorry, I could not find an answer in the PDF data.")
'''
The Question I Ask the LLM:
what is Schizophrenia Spectrum and how to prevent by this problem?

The Output is depand thatpdf have that informationor not.
'''