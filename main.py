import discord
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import pandas as pd
import asyncio
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

index_name = 'capstone-project'
index = pc.Index(index_name)

# Track ongoing conversations by channel
last_responses = {}

# Helper function to generate embeddings
async def get_embedding(text, retries=3, backoff_factor=5):
    """Generate embedding using OpenAI."""
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(backoff_factor * (2 ** attempt))
            else:
                raise e

# Function to delete outdated data from Pinecone
async def delete_from_pinecone(query):
    """Delete a vector from Pinecone based on its query."""
    try:
        vector_id = f"query-{hash(query)}"
        index.delete(ids=[vector_id])
        print(f"[DEBUG] Deleted outdated entry for query: {query}")
    except Exception as e:
        print(f"[ERROR] Failed to delete from Pinecone: {e}")

# Upsert corrections into Pinecone
async def upsert_to_pinecone(query, corrected_answer):
    """Upsert the updated query and response into Pinecone."""
    try:
        # Delete any outdated entries first
        await delete_from_pinecone(query)

        # Generate a new embedding for the query
        embedding = await get_embedding(query)
        if embedding is None:
            print("[ERROR] Skipping upsert due to missing embedding.")
            return

        # Upsert the updated query and response
        vector_id = f"query-{hash(query)}"
        metadata = {
            "OriginalQuery": query,
            "CorrectedAnswer": corrected_answer,
            "Timestamp": datetime.now(timezone.utc).isoformat()
        }

        index.upsert(vectors=[(vector_id, embedding, metadata)])
        print(f"[DEBUG] Upserted correction: Query: '{query}', Answer: '{corrected_answer}'")
    except Exception as e:
        print(f"[ERROR] Failed to upsert correction to Pinecone: {e}")

# Query Pinecone for the best match
async def find_correction_in_pinecone(query):
    """Retrieve the most relevant corrected response."""
    try:
        embedding = await get_embedding(query)
        result = index.query(vector=embedding, top_k=5, include_metadata=True)
        matches = result.get("matches", [])

        if matches and matches[0]["score"] > 0.85:
            return matches[0]["metadata"]["CorrectedAnswer"]
        return None
    except Exception as e:
        print(f"[ERROR] Failed to query Pinecone: {e}")
        return None

# Detect and process user corrections
async def process_correction(query, user_feedback):
    """Process user feedback to update or learn new corrections."""
    try:
        # Detect the updated response
        prompt = (
            f"Original Query: {query}\n"
            f"User Feedback: {user_feedback}\n\n"
            "Provide the updated response based on the feedback. If no changes are required, reply with 'No correction detected.'"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant processing corrections."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )

        corrected_response = response['choices'][0]['message']['content'].strip()
        if "No correction detected" in corrected_response:
            return None

        # Update Pinecone with the new response
        await upsert_to_pinecone(query, corrected_response)
        return corrected_response
    except Exception as e:
        print(f"[ERROR] Failed to process correction: {e}")
        return None

# Detect greetings
async def detect_greeting(user_message):
    """Detect if the user's message is a greeting."""
    predefined_greetings = ["hi", "hello", "hey", "greetings", "what's up", "sup"]
    if user_message.lower() in predefined_greetings:
        return "Hello! How can I assist you today?"

    try:
        prompt = f"Is the following text a greeting? '{user_message}'"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant detecting greetings."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        return "Hello! How can I assist you today?" if response['choices'][0]['message']['content'].strip().lower() == "yes" else None
    except Exception as e:
        print(f"[ERROR] Failed to detect greeting: {e}")
        return None

# Query OpenAI for a response
async def query_openai(user_message):
    """Generate a response using OpenAI for unmatched queries."""
    try:
        prompt = f"Answer the following question: {user_message}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content'].strip()
from bot import run_discord_bot
from responses import retrieve_documents, generate_response

def test_retrieval_and_response(query):
    try:
        # Retrieve documents using the query
        retrieved_docs = retrieve_documents(query)
        
        if retrieved_docs and 'matches' in retrieved_docs and len(retrieved_docs['matches']) > 0:
            # Generate a response based on retrieved documents and query
            generated_answer = generate_response(retrieved_docs, query)
            print(f"Query: {query}")
            print(f"Generated response: {generated_answer}")
        else:
            print(f"No relevant documents found for query: {query}")
    except Exception as e:
        print(f"[ERROR] Failed to query OpenAI: {e}")
        return None

# Discord bot setup
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

    user_message = message.content.strip()
    channel_id = message.channel.id

    try:
        # Step 1: Detect greetings
        greeting_response = await detect_greeting(user_message)
        if greeting_response:
            await message.channel.send(greeting_response)
            return

        # Step 2: Check for clarifications or expansions
        if channel_id in last_responses and user_message.lower() in ["expand", "clarify", "explain more"]:
            previous_query = last_responses[channel_id]
            existing_answer = await find_correction_in_pinecone(previous_query)
            if existing_answer:
                await message.channel.send(f"{existing_answer} Let me know if you'd like further details.")
                return

        # Step 3: Query Pinecone for an answer
        existing_answer = await find_correction_in_pinecone(user_message)
        if existing_answer:
            await message.channel.send(existing_answer)
            last_responses[channel_id] = user_message
            return

        # Step 4: Use OpenAI for unmatched queries
        openai_response = await query_openai(user_message)
        if openai_response:
            await message.channel.send(openai_response)
            last_responses[channel_id] = user_message
            return

        # Step 5: Fallback response
        await message.channel.send("I'm not sure about that. Could you provide more details?")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        await message.channel.send("An error occurred. Please try again.")

if __name__ == "__main__":
    client.run(TOKEN)
        print(f"An error occurred: {str(e)}")

# Test the document retrieval and response generation
if __name__ == '__main__':  
    queries = [
        "What is the startup you co-founded?",
        "How do I submit my project proposal?",
        "When is the final presentation due?"
    ]

    for query in queries:
        test_retrieval_and_response(query)
        print("-" * 50)  # Separator between queries

    # Uncomment the following line when you're ready to run the bot
    run_discord_bot()
