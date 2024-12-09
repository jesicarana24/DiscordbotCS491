import discord
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import asyncio
from datetime import datetime

load_dotenv()

# Initialize Pinecone and OpenAI API keys
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

index_name = 'capstone-project'
index = pc.Index(index_name)

async def get_embedding(text, retries=3, backoff_factor=60):
    """Generate embedding using OpenAI."""
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']
            if len(embedding) != 1536:
                raise ValueError(f"Embedding length is {len(embedding)}, but expected 1536.")
            return embedding
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            if "Incorrect API key" in str(e):
                # Stop retrying if the API key is invalid
                print("[ERROR] Invalid OpenAI API key. Please update the .env file with a valid key.")
                return None
            if attempt < retries - 1:
                wait = backoff_factor * (2 ** attempt)
                print(f"[DEBUG] Retrying in {wait} seconds...")
                await asyncio.sleep(wait)
            else:
                print("[ERROR] Maximum retries exceeded.")
                return None

async def upsert_to_pinecone(query, corrected_answer):
    """Upsert a corrected query and response into Pinecone."""
    embedding = await get_embedding(query)
    if embedding is None:
        print("[ERROR] Skipping upsert due to missing embedding.")
        return
    vector_id = f"correction-{hash(query)}"  # Use a consistent ID for the query
    metadata = {
        "CorrectedQuery": query,
        "CorrectedAnswer": corrected_answer,
        "Timestamp": datetime.utcnow().isoformat()
    }
    index.upsert(vectors=[(vector_id, embedding, metadata)])
    print(f"[DEBUG] Upserted correction: Query: '{query}', Answer: '{corrected_answer}'")

async def find_correction_in_pinecone(query):
    """Check Pinecone for a stored correction."""
    embedding = await get_embedding(query)
    if embedding is None:
        print("[ERROR] Skipping correction lookup due to missing embedding.")
        return None
    try:
        result = index.query(
            vector=embedding,
            top_k=3,  # Retrieve top matches
            include_metadata=True
        )
        corrections = [
            match["metadata"]
            for match in result["matches"]
            if "CorrectedAnswer" in match["metadata"]
        ]
        if corrections:
            # Sort by most recent timestamp and return the latest correction
            corrections.sort(key=lambda x: x["Timestamp"], reverse=True)
            print(f"[DEBUG] Latest correction retrieved: {corrections[0]['CorrectedAnswer']}")
            return corrections[0]["CorrectedAnswer"]
        print("[DEBUG] No correction found in Pinecone.")
    except Exception as e:
        print(f"[ERROR] Failed to query Pinecone for corrections: {e}")
    return None

TOKEN = os.getenv('TOKEN')
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"[INFO] {client.user} is now running!")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_message = str(message.content).lower()

    try:
        # Handle corrections
        if "it's actually" in user_message or "no, that's wrong" in user_message:
            if message.reference and message.reference.message_id:
                original_query = message.reference.resolved.content.strip()
                corrected_answer = user_message.split("it's actually")[-1].strip()
                await upsert_to_pinecone(original_query, corrected_answer)
                await message.channel.send("Thanks for the correction! I've updated my memory.")
            else:
                await message.channel.send("Please reply to the message you want to correct.")
            return

        # Check for corrections in Pinecone
        correction = await find_correction_in_pinecone(user_message)
        if correction:
            await message.channel.send(correction)
            return

        # Default greeting
        if user_message in ["hey", "hello", "hi"]:
            await message.channel.send("Hello! How can I help you today?")
            return

        # Fallback to OpenAI or Pinecone retrieval if no correction found
        try:
            embedding = await get_embedding(user_message)
            if embedding is None:
                await message.channel.send("An error occurred while processing your request.")
                return
            result = index.query(vector=embedding, top_k=3, include_metadata=True)
            response = generate_response(result, user_message)
            await message.channel.send(response)
        except Exception as e:
            print(f"[ERROR] Failed to generate response: {e}")
            await message.channel.send(f"An error occurred: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        await message.channel.send("An unexpected error occurred. Please try again later.")

def generate_response(result, query):
    """Generate a response based on Pinecone results."""
    combined_docs = " ".join([
        f"Question: {doc['metadata'].get('Question', 'N/A')} Answer: {doc['metadata'].get('Answer', 'N/A')}"
        for doc in result["matches"]
    ])
    prompt = f"Based on the following information: {combined_docs}, answer the question: {query}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[ERROR] Failed to generate response from OpenAI: {e}")
        raise

if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except Exception as e:
        print(f"[ERROR] Bot failed to start: {e}")
