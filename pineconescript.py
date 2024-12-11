
import time
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')  
df = pd.read_csv('organized_faq_data.csv')
small_df = df[df['Difficulty'] == 1]

if small_df.empty:
    raise Exception("No rows found with Difficulty == 1.")

small_df['combined'] = (
    small_df['Question'] + " " +
    small_df['Answer'].fillna('') + " " +
    small_df['Context'].fillna('') + " " +
    small_df['Intent'].fillna('') + " " +
    small_df['Entities'].fillna('') + " " +
    small_df['Comments'].fillna('')
)

def get_embedding(text, retries=3, backoff_factor=60):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']

          
            if len(embedding) != 1536:
                raise ValueError(f"Embedding length is {len(embedding)}, but expected 1536.")

            print(f"Generated embedding length: {len(embedding)}")  
            return embedding
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < retries - 1:
                wait = backoff_factor * (2 ** attempt)
                print(f"Waiting for {wait} seconds before retrying...")
                time.sleep(wait)
            else:
                print("Maximum retries exceeded.")
                raise


embeddings = []
for text in small_df['combined']:
    try:
        embedding = get_embedding(text)
        embeddings.append(embedding)
        print(f"Generated embedding for: {text[:50]}...")  
    except Exception as e:
        print(f"Failed to generate embedding for: {text[:50]}... - {e}")

if not embeddings:
    raise Exception("No embeddings were generated due to errors.")

# Ensure all embeddings have correct dimensions
final_embeddings = []
for embed in embeddings:
    if len(embed) == 1536:  # Check if each embedding is of the correct size
        final_embeddings.append([float(e) for e in embed])  # Ensure all values are floats
    else:
        print(f"Invalid embedding size: {len(embed)}, skipping...")
# Proceed with upsert if valid embeddings exist
if final_embeddings:
    index_name = 'capstone-project'
    
    # Get the index instance
    index = pc.Index(index_name)

    # Specify the namespace
    namespace = "your_namespace"

    # Prepare data for upsert with metadata
    vectors_to_upsert = [
        (
            f'row-{i}',  # Unique ID
            final_embeddings[i],  # The embedding
            {  # Metadata for each vector, using fallback values if needed
                'Difficulty': float(small_df.iloc[i]['Difficulty']),
                'Volatility': float(small_df.iloc[i]['Volatility Level']),
                'question': small_df.iloc[i]['Question'] or 'Unknown question',
                'answer': small_df.iloc[i]['Answer'] or 'Unknown answer'
            }
        )
        for i in range(len(final_embeddings))
    ]

    # Perform the upsert operation with a namespace
    index.upsert(vectors=vectors_to_upsert, namespace=namespace)

    print(f"Upserted {len(vectors_to_upsert)} vectors to the index '{index_name}' in namespace '{namespace}'.")
