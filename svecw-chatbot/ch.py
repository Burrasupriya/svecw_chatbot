import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

# Load environment variables
load_dotenv()

# Apply asyncio fix
nest_asyncio.apply()

# Get API keys
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate API keys
if not all([LLAMA_CLOUD_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise ValueError("Missing one or more API keys. Please check your .env file.")

# Initialize LlamaParse for counselling data
parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")

# Read Excel File
file_path = "./data/grouped_data.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please check the path.")

file_extractor = {".xlsx": parser}
documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()

# Initialize Groq LLM
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.3)
Settings.llm = llm

# Initialize Embedding Model
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Initialize Qdrant Client
qdrantClient = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY)

# Create Vector Store
vector_store = QdrantVectorStore(client=qdrantClient, collection_name="counselling_data")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create and store vector store index
VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Load the index from storage
db_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Create a query engine
query_engine = db_index.as_query_engine()

# Function to get response
def get_response(query):
    response = query_engine.query(query)
    return response.response


# Interactive query session
if __name__ == "__main__":
    print("Ask questions about the counselling data. Type 'exit' to quit.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            break
        response = get_response(query)
        print("Response:", response)
