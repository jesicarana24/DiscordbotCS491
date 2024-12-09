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
                print("[ERROR] Invalid OpenAI API key. Please update the .env file.")
                return None
            if attempt < retries - 1:
                wait = backoff_factor * (2 ** attempt)
                print(f"[DEBUG] Retrying in {wait} seconds...")
                await asyncio.sleep(wait)
            else:
                print("[ERROR] Maximum retries exceeded.")
                return None

async def upsert_to_pinecone(query, corrected_answer):
    """Upsert or overwrite a corrected query and response into Pinecone."""
    try:
        embedding = await get_embedding(query)
        if embedding is None:
            print("[ERROR] Skipping upsert due to missing embedding.")
            return

        # Use the query itself as the unique vector ID to overwrite existing data
        vector_id = f"query-{hash(query)}"
        metadata = {
            "OriginalQuery": query,
            "CorrectedAnswer": corrected_answer,
            "Timestamp": datetime.utcnow().isoformat()
        }

        # Upsert the correction
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        print(f"[DEBUG] Upserted correction: Query: '{query}', Answer: '{corrected_answer}'")
    except Exception as e:
        print(f"[ERROR] Failed to upsert correction to Pinecone: {e}")

async def find_correction_in_pinecone(query):
    """Retrieve the corrected response for a query from Pinecone."""
    try:
        embedding = await get_embedding(query)
        if embedding is None:
            print("[ERROR] Skipping correction lookup due to missing embedding.")
            return None

        # Query Pinecone for top matches
        result = index.query(
            vector=embedding,
            top_k=1,  # Fetch the most relevant match
            include_metadata=True
        )

        if result and result["matches"]:
            # Fetch the metadata of the best match
            match = result["matches"][0]
            if "CorrectedAnswer" in match["metadata"]:
                corrected_answer = match["metadata"]["CorrectedAnswer"]
                print(f"[DEBUG] Retrieved corrected answer: {corrected_answer}")
                return corrected_answer

        print("[DEBUG] No correction found in Pinecone.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to query Pinecone for corrections: {e}")
        return None

async def detect_correction(original_response, user_message):
    """Use ChatGPT to detect if a user message is feedback/correction."""
    try:
        prompt = (
            f"Original Response: {original_response}\n"
            f"User Feedback: {user_message}\n\n"
            "Is the user's message a correction or feedback? If so, identify what needs to be corrected "
            "and provide the updated response. Otherwise, say 'No correction needed.'"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that identifies corrections in feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        corrected_data = response['choices'][0]['message']['content'].strip()
        return corrected_data
    except Exception as e:
        print(f"[ERROR] Failed to detect correction: {e}")
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
    """Handle incoming messages."""
    if message.author == client.user:
        return

    user_message = message.content.strip()

    try:
        # If the message is a reply, check for corrections
        if message.reference and message.reference.message_id:
            original_query = message.reference.resolved.content.strip()

            # Use ChatGPT to detect corrections
            prompt = (
                f"Original Response: {original_query}\n"
                f"User Feedback: {user_message}\n\n"
                "Is the user's message a correction or feedback? If so, provide the corrected response. "
                "If it is not a correction, say 'No correction needed.'"
            )
            chat_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that identifies corrections in feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            corrected_data = chat_response['choices'][0]['message']['content'].strip()

            if corrected_data and "No correction needed" not in corrected_data:
                # Update Pinecone with the new correction
                await upsert_to_pinecone(original_query, corrected_data)
                await message.channel.send("Thanks for the correction! I've updated my memory.")
            else:
                await message.channel.send("I couldn't detect any corrections. Please clarify.")
            return

        # Always retrieve the latest correction for a query
        correction = await find_correction_in_pinecone(user_message)
        if correction:
            await message.channel.send(correction)
            return

        # Default response
        if user_message.lower() in ["hey", "hello", "hi"]:
            await message.channel.send("Hello! How can I help you today?")
            return

        # Fallback to OpenAI or standard Pinecone response
        embedding = await get_embedding(user_message)
        if embedding is None:
            await message.channel.send("An error occurred while processing your request.")
            return

        result = index.query(vector=embedding, top_k=3, include_metadata=True)
        response = generate_response(result, user_message)
        await message.channel.send(response)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        await message.channel.send("An unexpected error occurred. Please try again later.")

def generate_response(result, query):
    """Generate a response based on Pinecone results."""
    combined_docs = " ".join([
        f"Question: {doc['metadata'].get('OriginalQuery', 'N/A')} Answer: {doc['metadata'].get('CorrectedAnswer', 'N/A')}"
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
