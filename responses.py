import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Create or connect to Pinecone index
index_name = 'capstone-project'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches your embedding size
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Function to normalize embedding
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    normalized_embedding = embedding / norm
    clamped_embedding = np.clip(normalized_embedding, -0.9, 0.9)
    rounded_embedding = np.round(clamped_embedding, 6)
    return rounded_embedding.tolist()

# Retrieve documents from Pinecone
# Retrieve documents from Pinecone
# Retrieve documents from Pinecone
def retrieve_documents(query):
    try:
        # Get embedding from OpenAI
        response = openai.Embedding.create(input=[query], model="text-embedding-ada-002")
        query_embedding = response['data'][0]['embedding']

        # Ensure that the embedding length is correct
        if len(query_embedding) != 1536:
            raise ValueError(f"Embedding length is {len(query_embedding)}, but expected 1536.")

        # Normalize the embedding before querying Pinecone
        query_embedding = normalize_embedding(query_embedding)

        # Query Pinecone with the generated embedding using keyword arguments
        result = index.query(
            vector=query_embedding,  # This is the query vector
            top_k=10,                 # Number of results you want to retrieve
            namespace=''              # Namespace, leave empty if not using any
        )


        # Optionally describe the index stats for debugging
        index.describe_index_stats()

        return result
    except Exception as e:
        print(f"An error occurred during document retrieval: {e}")
        raise

# Handle user response
def handle_response(user_message):
    try:
        # Generate an embedding for the user message
        embedding = openai.Embedding.create(input=user_message, model="text-embedding-ada-002")["data"][0]["embedding"]
        return "Response based on the message", embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return "Sorry, an error occurred.", None

# Generate response using ChatCompletion
# Generate response using ChatCompletion
# Generate response using ChatCompletion
def generate_response(retrieved_docs, query):
    combined_docs = []

    for doc in retrieved_docs['matches']:
        # Check if 'metadata' is present
        if 'metadata' in doc:
            question = doc['metadata'].get('question', 'Unknown question')
            answer = doc['metadata'].get('answer', 'Unknown answer')
            combined_docs.append(f"Q: {question} A: {answer}")
        else:
            combined_docs.append("No metadata available for this document.")
    
    combined_docs_str = " ".join(combined_docs)
    
    prompt = f"Based on the following information: {combined_docs_str}, answer the question: {query}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    return response.choices[0].message['content'].strip()

# Example usage
query_text = "What industry projects are available?"
retrieved_docs = retrieve_documents(query_text)
response = generate_response(retrieved_docs, query_text)
print(response)
