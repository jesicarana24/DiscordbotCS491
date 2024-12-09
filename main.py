import discord
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import time
from datetime import datetime

load_dotenv()

# Initialize Pinecone and OpenAI API keys
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

index_name = 'capstone-project'
index = pc.Index(index_name)

def get_embedding(text, retries=3, backoff_factor=60):
    """Generate embedding using OpenAI."""
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']
            if len(embedding) != 1536:
                raise ValueError(f"Embedding length is {len(embedding)}, but expected 1536.")
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

def query_pinecone(query_embedding):
    """Query Pinecone with an embedding."""
    query_vector = [float(e) for e in query_embedding]
    result = index.query(
        vector=query_vector,
        top_k=3,
        include_values=True,
        include_metadata=True
    )
    return result

def upsert_to_pinecone(query, corrected_answer):
    """Upsert a corrected query and response into Pinecone."""
    embedding = get_embedding(query)
    timestamp = datetime.utcnow().isoformat()  # Add timestamp for prioritization
    metadata = {
        "CorrectedQuery": query,
        "CorrectedAnswer": corrected_answer,
        "Timestamp": timestamp
    }
    vector_id = f"correction-{hash(query)}"  # Use a hash of the query as a unique ID
    index.upsert(vectors=[(vector_id, embedding, metadata)])
    print(f"Upserted correction: Query: '{query}', Answer: '{corrected_answer}', Timestamp: '{timestamp}'")

def find_latest_correction_in_pinecone(query):
    """Check Pinecone for the latest stored correction."""
    embedding = get_embedding(query)
    result = query_pinecone(embedding)

    # Sort matches by timestamp (most recent first)
    corrections = [
        match["metadata"]
        for match in result["matches"]
        if "CorrectedAnswer" in match["metadata"]
    ]
    if corrections:
        corrections.sort(key=lambda x: x.get("Timestamp", ""), reverse=True)
        return corrections[0]["CorrectedAnswer"]  # Return the most recent correction
    return None

def generate_response(retrieved_docs, query):
    """Generate a response based on retrieved documents."""
    combined_docs = " ".join([
        f"Question: {doc['metadata'].get('Question', 'N/A')} Answer: {doc['metadata'].get('Answer', 'N/A')}"
        for doc in retrieved_docs['matches']
    ])
    prompt = f"Based on the following information: {combined_docs}, answer the question: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps answer questions based on retrieved documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

TOKEN = os.getenv('TOKEN')
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} is now running!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_message = str(message.content).lower()

    # Check for corrections
    if "it's actually" in user_message or "no, that's wrong" in user_message:
        if message.reference and message.reference.message_id:
            original_query = message.reference.resolved.content  # Get the original question text
            corrected_answer = user_message.split("it's actually")[-1].strip()  # Extract corrected answer
            # Save the corrected answer to Pinecone
            upsert_to_pinecone(original_query, corrected_answer)
            await message.channel.send("Thanks for the correction! I've updated my memory.")
        else:
            await message.channel.send("Please reply to the message you want to correct.")
        return

    # Check Pinecone for the latest relevant correction
    correction = find_latest_correction_in_pinecone(user_message)
    if correction:
        await message.channel.send(correction)
        return

    if user_message in ["hey", "hello", "hi"]:
        await message.channel.send("Hello! How can I help you today?")
        return

    try:
        query_embedding = get_embedding(user_message)
        retrieved_docs = query_pinecone(query_embedding)
        response = generate_response(retrieved_docs, user_message)
        await message.channel.send(response)
    except Exception as e:
        await message.channel.send(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    client.run(TOKEN)
