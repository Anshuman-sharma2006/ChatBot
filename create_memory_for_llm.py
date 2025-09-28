from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("Environment variables loaded.", flush=True)

# -------------------------------
# Step 1: Load raw PDF(s)
# -------------------------------
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
print("STEP 1: Loading PDFs...", flush=True)
documents = load_pdf_files(data=DATA_PATH)
print("✅ Documents loaded:", len(documents), flush=True)

# -------------------------------
# Step 2: Create Chunks
# -------------------------------
#Create text chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks
print("STEP 2: Creating chunks...", flush=True)
text_chunks = create_chunks(extracted_data=documents)
print("✅ Chunks created:", len(text_chunks), flush=True)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
print("STEP 3: Loading embedding model (this may take a while)...", flush=True)
embedding_model=get_embedding_model()
print("✅ Embedding model ready", flush=True)

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
print("STEP 4: Creating FAISS DB...", flush=True)
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print("✅ FAISS DB saved at:", DB_FAISS_PATH, flush=True)
# -------------------------------
# Step 5: Load embeddings from FAISS
# -------------------------------
db = FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)
print("FAISS vectorstore loaded successfully!", flush=True)