import os
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))

# Define index name and dimensions
index_name = 'capstone-project'
embedding_dimension = 1536

# Check if index exists, if so, delete and recreate
if index_name in pinecone.list_indexes():
    print(f"Index '{index_name}' already exists. Deleting and recreating...")
    
    pinecone.delete_index(index_name)

# Create the index with the proper dimensions
pinecone.create_index(
    name=index_name,
    dimension=embedding_dimension,
    metric='cosine'  # You can change this depending on the use case
)

print(f"Index '{index_name}' created successfully with dimension {embedding_dimension}.")
