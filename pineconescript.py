# import time
# import os
# import openai
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# import pandas as pd

# # Load environment variables
# load_dotenv()

# # Set pandas display options for easier debugging
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

# # Initialize Pinecone and OpenAI clients
# pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
# openai.api_key = os.getenv('OPENAI_API_KEY')  # Initialize the OpenAI client with the correct API key

# # Load and filter the CSV data
# df = pd.read_csv('organized_faq_data.csv')
# small_df = df[df['Difficulty'] == 1]

# if small_df.empty:
#     raise Exception("No rows found with Difficulty == 1.")

# # Sample one row for simplicity (modify to sample more rows as needed)
# small_df = small_df.sample(n=20)

# # Create a combined column for embedding input
# small_df['combined'] = (
#     small_df['Question'] + " " +
#     small_df['Answer'].fillna('') + " " +
#     small_df['Context'].fillna('') + " " +
#     small_df['Intent'].fillna('') + " " +
#     small_df['Entities'].fillna('') + " " +
#     small_df['Comments'].fillna('')
# )

# # Function to retrieve the embedding with retries
# def get_embedding(text, retries=3, backoff_factor=60):
#     for attempt in range(retries):
#         try:
#             response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
#             embedding = response['data'][0]['embedding']

#             # Ensure the embedding length is 1536
#             if len(embedding) != 1536:
#                 raise ValueError(f"Embedding length is {len(embedding)}, but expected 1536.")

#             print(f"Generated embedding length: {len(embedding)}")  # For debugging
#             return embedding
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             if attempt < retries - 1:
#                 wait = backoff_factor * (2 ** attempt)
#                 print(f"Waiting for {wait} seconds before retrying...")
#                 time.sleep(wait)
#             else:
#                 print("Maximum retries exceeded.")
#                 raise

# # Generate embeddings
# embeddings = []
# for text in small_df['combined']:
#     try:
#         embedding = get_embedding(text)
#         embeddings.append(embedding)
#         print(f"Generated embedding for: {text[:50]}...")  # Print first 50 characters for debugging
#     except Exception as e:
#         print(f"Failed to generate embedding for: {text[:50]}... - {e}")

# # Check if embeddings are generated
# if not embeddings:
#     raise Exception("No embeddings were generated due to errors.")

# # Prepare embeddings for upsert and include metadata (like question and answer)
# final_embeddings = [[float(e) for e in embed] for embed in embeddings]

# # Proceed if embeddings are generated
# if final_embeddings:
#     index_name = 'capstone-project'
#     # Get the index instance
#     index = pc.Index(index_name)

#     # Prepare data for upsert (with metadata)
#     vectors_to_upsert = [
#         (
#             f'row-{i}',  # Unique ID
#             final_embeddings[i],  # The embedding
#             {  # Metadata for each vector
#                 'Difficulty': float(small_df.iloc[i]['Difficulty']),
#                 'Volatility': float(small_df.iloc[i]['Volatility Level']),
#                 'Question': small_df.iloc[i]['Question'],  # Include the question as metadata
#                 'Answer': small_df.iloc[i]['Answer'],      # Include the answer as metadata
#                 'Context': small_df.iloc[i]['Context'],    # Include the context as metadata
#                 'CombinedText': small_df.iloc[i]['combined']  # Include the combined text as metadata
#             }
#         )
#         for i in range(len(final_embeddings))
#     ]

#     # Perform the upsert operation with metadata
#     index.upsert(vectors=vectors_to_upsert)
#     print(f"Upserted {len(vectors_to_upsert)} vectors to the index '{index_name}'.")

# else:
#     raise Exception("No embeddings available to create or update the index.")

# print("Script completed successfully.")
import time
import os
import openai
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
openai.api_key = os.getenv('OPENAI_API_KEY')  # Initialize the OpenAI client with the correct API key

# Load and filter the CSV data
df = pd.read_csv('organized_faq_data.csv')
small_df = df[df['Difficulty'] == 1]

if small_df.empty:
    raise Exception("No rows found with Difficulty == 1.")

# Remove sampling to process the entire dataset
# Create a combined column for embedding input
small_df['combined'] = (
    small_df['Question'] + " " +
    small_df['Answer'].fillna('') + " " +
    small_df['Context'].fillna('') + " " +
    small_df['Intent'].fillna('') + " " +
    small_df['Entities'].fillna('') + " " +
    small_df['Comments'].fillna('')
)

# Function to retrieve the embedding with retries
def get_embedding(text, retries=3, backoff_factor=60):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']

            # Ensure the embedding length is 1536
            if len(embedding) != 1536:
                raise ValueError(f"Embedding length is {len(embedding)}, but expected 1536.")

            print(f"Generated embedding length: {len(embedding)}")  # For debugging
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
    raise Exception("No embeddings were generated due to errors.")

# Prepare embeddings for upsert and include metadata (like question and answer)
final_embeddings = [[float(e) for e in embed] for embed in embeddings]

# Proceed if embeddings are generated
if final_embeddings:
    index_name = 'capstone-project'
    # Get the index instance
    index = pc.Index(index_name)

    # Prepare data for upsert (with metadata)
    vectors_to_upsert = [
        (
            f'row-{i}',  # Unique ID
            final_embeddings[i],  # The embedding
            {  # Metadata for each vector, all converted to strings and with missing values handled
                'Difficulty': str(small_df.iloc[i]['Difficulty']),  # Convert to string
                'Volatility': str(small_df.iloc[i].get('Volatility Level', '')),  # Handle missing values
                'Question': str(small_df.iloc[i]['Question']),  # Ensure it's a string
                'Answer': str(small_df.iloc[i]['Answer']),
                'Context': str(small_df.iloc[i]['Context']),
                'CombinedText': str(small_df.iloc[i]['combined'])
            }
        )
        for i in range(len(final_embeddings))
    ]

    # Perform the upsert operation with metadata
    index.upsert(vectors=vectors_to_upsert)
    print(f"Upserted {len(vectors_to_upsert)} vectors to the index '{index_name}'.")

else:
    raise Exception("No embeddings available to create or update the index.")

print("Script completed successfully.")
