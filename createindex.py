import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = 'capstone-project-jesica-2'
embedding_dimension = 1536

if index_name in pc.list_indexes():
    print(f"Index '{index_name}' already exists. Deleting the existing index to recreate it.")
    pc.delete_index(index_name)

print(f"Creating a new index '{index_name}' with dimension {embedding_dimension}...")
pc.create_index(
    name=index_name,
    dimension=embedding_dimension,
    metric='cosine',
    spec=ServerlessSpec(
        cloud="aws",
        region=os.getenv('PINECONE_ENV')
    )
)

print(f"Index '{index_name}' created successfully with dimension {embedding_dimension}.")
