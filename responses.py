import openai
import pinecone
import os
import csv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import openai
import pinecone

# Initialize OpenAI (Assuming Pinecone is already initialized in another script)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Connect to the Pinecone index (index already created in pineconescript.py)
index = pinecone.Index('capstone-project')

# Function to retrieve relevant documents from Pinecone
def retrieve_documents(query):
    # Generate an embedding for the query
    query_embedding = openai.Embedding.create(input=query, model='text-embedding-ada-002')['data'][0]['embedding']
    
    # Query Pinecone for the closest matches
    result = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
    
    return result

# Function to generate a response using OpenAI (or any other LLM)
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

# Function to handle the response logic
def handle_response(message) -> str:
    # Lowercase the message for easier handling
    p_message = message.lower()

    # Basic responses for specific inputs
    if p_message == 'hello':
        return 'Hey There!'
    
    if p_message == 'roll':
        return str(random.randint(1, 6))
    
    if p_message == '!help':
        return "`This is a help message that you can modify.`"
    
    # Default behavior: use RAG model to find the answer
    retrieved_docs = retrieve_documents(p_message)
    
    if len(retrieved_docs['matches']) == 0:
        return "I don't know what you said, babe <3"

    # Generate a meaningful response based on retrieved documents
    generated_answer = generate_response(retrieved_docs, p_message)
    
    return generated_answer
