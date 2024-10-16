import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone using the new API
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Connect to the existing index
index_name = 'capstone-project'
index = pc.Index(index_name)

# Function to retrieve relevant documents from Pinecone
def retrieve_documents(query):
    # Generate an embedding for the query using the correct OpenAI method
    response = openai.Embedding.create(input=query, model='text-embedding-ada-002')
    query_embedding = response['data'][0]['embedding']
    
    # Query Pinecone for the closest matches
    result = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
    
    return result

# Function to generate a response using OpenAI
def generate_response(retrieved_docs, query):
    # Combine the 'question' and 'answer' from the retrieved documents to form a context
    combined_docs = " ".join([doc['metadata']['question'] + ' ' + doc['metadata']['answer'] for doc in retrieved_docs['matches']])
    
    # Create a prompt with the context and the user query
    prompt = f"Based on the following information: {combined_docs}, answer the question: {query}"
    
    # Generate a response from OpenAI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    
    return response.choices[0].text.strip()
