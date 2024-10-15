import time
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Set pandas display options for easier debugging
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Initialize Pinecone and OpenAI clients
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load and filter the CSV data
df = pd.read_csv('organized_faq_data.csv')
small_df = df[df['Difficulty'] == 1]

if small_df.empty:
    raise Exception("No rows found with Difficulty == 1.")

# Sample one row for simplicity
small_df = small_df.sample(n=1)

# Create a combined column for embedding input
small_df['combined'] = (
    small_df['Question'] + " " +
    small_df['Answer'].fillna('') + " " +
    small_df['Context'].fillna('') + " " +
    small_df['Intent'].fillna('') + " " +
    small_df['Entities'].fillna('') + " " +
    small_df['Comments'].fillna('')
)

# Define a function to retrieve the embedding with retries
def get_embedding(text, retries=3, backoff_factor=60):
    for attempt in range(retries):
        try:
            response = client.embeddings.create(input=text, model='text-embedding-ada-002')
            return response.data[0].embedding
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < retries - 1:
                wait = backoff_factor * (2 ** attempt)
                print(f"Waiting for {wait} seconds before retrying...")
                time.sleep(wait)
            else:
                print("Maximum retries exceeded.")
                raise

# Generate embeddings
embeddings = []
for text in small_df['combined']:
    try:
        embedding = get_embedding(text)
        embeddings.append(embedding)
        print(f"Generated embedding for: {text[:50]}...")  # Print first 50 characters for debugging
    except Exception as e:
        print(f"Failed to generate embedding for: {text[:50]}... - {e}")

# Check if embeddings are generated
if not embeddings:
    raise Exception("No embeddings were generated due to quota exhaustion or errors.")

# Add additional columns to embeddings
final_embeddings = []
for i, embed in enumerate(embeddings):
    # Convert numpy float64 to regular Python float
    embed = [float(e) for e in embed]  # Convert each embedding element to a Python float
    extended_embedding = embed + [float(small_df.iloc[i]['Difficulty']), float(small_df.iloc[i]['Volatility Level'])]
    final_embeddings.append(extended_embedding)

# Proceed if embeddings are generated
if final_embeddings:
    index_name = 'capstone-project'

    # Check if the index exists, skip creation if it does
    if index_name not in pc.list_indexes():
        print(f"Creating a new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1538,  # Ensure this matches the actual size of your final embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        print(f"Index '{index_name}' already exists. Skipping index creation.")

    #Get the index instance
    index = pc.Index(index_name)

    # Prepare data for upsert (using unique row IDs)
    ids = [f'row-{i}' for i in range(len(final_embeddings))]
    vectors_to_upsert = list(zip(ids, final_embeddings))

    # Perform the upsert operation
    index.upsert(vectors=vectors_to_upsert)
    print(f"Upserted {len(vectors_to_upsert)} vectors to the index '{index_name}'.")

else:
    raise Exception("No embeddings available to create or update the index.")

print("Script completed successfully.")
