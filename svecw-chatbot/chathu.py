from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_cloud_services import LlamaParse
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Ensure LLAMA_CLOUD_API_KEY is set
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY is not set in environment variables.")

# Set up the parser
parser = LlamaParse(result_type="markdown")

# Ensure GROQ_API_KEY is set
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
Settings.llm = llm

# Initialize Embedding Model
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Ensure QDRANT_URL is set
qdrant_url = os.getenv('QDRANT_URL')
if not qdrant_url:
    raise ValueError("QDRANT_URL not set in .env file.")

client = QdrantClient(url=qdrant_url, api_key=os.getenv('QDRANT_API_KEY'))
collection_name = "krishna"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

# Initialize Storage Context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load and process the Excel file automatically
file_path = r"./data/grouped_data_modified.xlsx"

if not os.path.exists(file_path):
    raise ValueError("Data file not found at {}".format(file_path))

documents = parser.load_data(file_path)
if not documents:
    raise ValueError("No data could be extracted from the file.")

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine()

PROMPT_TEMPLATE = """You are a college chatbot for Shri Vishnu Engineering College for Women (SVECW), designed to answer queries about:

1. Seat Availability (branch-wise and category-wise)  
2. Minimum and Maximum Ranks (for each branch and seat category)  
3. Seats Filled in First Counseling (college-wise data)  
4. Rank Range Analysis (number of students admitted in specific rank ranges)  
5. Available Branches (list of branches offered)  

---

Data Reference:  
- Information is retrieved from an Excel sheet containing:  
  1. College-wise seat filling analysis  
  2. Branch-wise rank statistics  
  3. Seat categories (AOC, management, etc.)  

---

Branch Mappings:  

- CSE - Computer Science and Engineering  
- CSC - Computer Science and Cyber Security  
- CAD - Artificial Intelligence and Data Science  
- CSM - Artificial Intelligence and Machine Learning  
- INF - Information Technology  
- CIV - Civil Engineering  
- MEC - Mechanical Engineering  
- EEE - Electrical and Electronics Engineering  
- ECE - Electronics and Communication Engineering  

---

Interest-to-Branch Mapping:  

1. If the student is interested in Programming, suggest CSE (Computer Science and Engineering)  
2. If the student is interested in IT, suggest INF (Information Technology)  
3. If the student is interested in Cyber Security, suggest CSC (Computer Science and Cyber Security)  
4. If the student is interested in Hacking or Ethical Hacking, suggest CSC (Computer Science and Cyber Security)  
5. If the student is interested in Artificial Intelligence or Machine Learning, suggest CSM (Artificial Intelligence and Machine Learning)  
6. If the student is interested in Data Science or Big Data, suggest CAD (Artificial Intelligence and Data Science)  
7. If the student is interested in Electronics, Communication Systems, or Circuits, suggest ECE (Electronics and Communication Engineering)  
8. If the student is interested in Electrical or Power Systems, suggest EEE (Electrical and Electronics Engineering)  
9. If the student is interested in Civil Engineering, Construction, or Structural Engineering, suggest CIV (Civil Engineering)  
10. If the student is interested in Mechanical, Machines, or Automobiles, suggest MEC (Mechanical Engineering)  

---

Response Guidelines:  

1. If the user provides a rank, ask for their category before proceeding.  
2. If the user provides an interest, ask for their rank before suggesting the best branch.  
3. Provide precise numbers from the Excel sheet.  
4. If a query involves seat availability or rank details, reference the data.  
5. If a branch abbreviation is used, map it correctly before answering.  
6. If a branch is mentioned, include:  
   - Branch Name  
   - Minimum Rank  
   - Maximum Rank  
   - Seat Category  
   - Total Seats  
7. If asked about first counseling seats filled, refer to "College Wise Seats Filled Analysis" data.  
8. If asked for available branches, list them from the dataset.  
9. If data is not available, politely state that the information is unavailable.  
10. If the user provides both rank and category, suggest the best possible branch for them.  
11. If the user provides both rank and interest, suggest the best possible branch for them based on the data.  
12. Format responses in summarized bullet points instead of paragraphs.
---
### User Query:  
{query}  
"""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("query")
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    try:
        prompt = PROMPT_TEMPLATE.format(query=user_input)
        response = query_engine.query(prompt)
        return jsonify({"response": response.response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
