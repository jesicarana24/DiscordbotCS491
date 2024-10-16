import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Define the index name and the embedding dimension
index_name = 'capstone-project'
embedding_dimension = 1536  # This should match the dimension of the OpenAI embeddings

# Check if the index already exists
if index_name in pc.list_indexes():
    print(f"Index '{index_name}' already exists. Deleting the existing index to recreate it.")
    pc.delete_index(index_name)

# Create the new index with the correct dimension
print(f"Creating a new index '{index_name}' with dimension {embedding_dimension}...")
pc.create_index(
    name=index_name,
    dimension=embedding_dimension,  # Make sure this matches the embedding size
    metric='cosine',  # You can also use 'euclidean' depending on your use case
    spec=ServerlessSpec(
        cloud="aws",  # You can also change this to 'gcp' if using Google Cloud
        region=os.getenv('PINECONE_ENV')  # Ensure the region matches the one you want to use
    )
)

print(f"Index '{index_name}' created successfully with dimension {embedding_dimension}.")
