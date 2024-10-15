import time
import openai
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

openai.api_key = os.getenv('OPENAI_API_KEY')

df = pd.read_csv('organized_faq_data.csv')

small_df = df[df['Difficulty'] == 1]  # Filter first
if small_df.empty:
    raise Exception("No rows found with Difficulty == 1.")

small_df = df.sample(n=1)

if small_df.empty:
    raise Exception("The filtered dataset is empty. Please check the filtering conditions.")

small_df['combined'] = (
    small_df['Question'] + " " +
    small_df['Answer'].fillna('') + " " +
    small_df['Context'].fillna('') + " " +
    small_df['Intent'].fillna('') + " " +
    small_df['Entities'].fillna('') + " " +
    small_df['Comments'].fillna('')
)


def get_embedding(text):
    retries = 3
    backoff_factor = 60  # Wait time in seconds

    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            return response['data'][0]['embedding']
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
            if attempt < retries - 1:
                wait = backoff_factor * (2 ** attempt)  # Exponential backoff
                print(f"Waiting for {wait} seconds before retrying...")
                time.sleep(wait)
            else:
                print("Maximum retries exceeded.")
                raise
        except openai.error.OpenAIError as e:
            print(f"An OpenAI API error occurred: {e}")
            raise
embeddings = []
for text in small_df['combined']:
    try:
        embedding = get_embedding(text)
        embeddings.append(embedding)
        print(f"Generated embedding for: {text}")  # Debug print
    except Exception as e:
        print(f"Failed to generate embedding for: {text} - {e}")

# Early check for empty embeddings
if not embeddings:
    raise Exception("No embeddings were generated due to quota exhaustion or errors.")

# Ensure that embeddings exist before processing
final_embeddings = []
for i, embed in enumerate(embeddings):
    extended_embedding = embed + [small_df.iloc[i]['Difficulty'], small_df.iloc[i]['Volatility Level']]
    final_embeddings.append(extended_embedding)

# Avoid IndexError: Ensure there are embeddings before accessing
if final_embeddings:
    index_name = 'capstone-project'
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=len(final_embeddings[0]),  # Adjust dimension to match the embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)

    ids = [f'row-{i}' for i in range(len(final_embeddings))]

    vectors_to_upsert = list(zip(ids, final_embeddings))

    index.upsert(vectors=vectors_to_upsert)
else:
    raise Exception("No embeddings available to create the index.")
