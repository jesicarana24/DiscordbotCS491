import discord
import os
import numpy as np
from dotenv import load_dotenv
from responses import handle_response
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Pinecone client using the new method
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Define the index name
index_name = 'capstone-project'

# Check if the index exists and create one if needed
if index_name not in pc.list_indexes().names():
    print(f"Creating a new index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # Match the embedding size
        metric='cosine',  # or 'euclidean', depending on your use case
        spec=ServerlessSpec(
            cloud='aws',
            region=os.getenv('PINECONE_ENV')  # Ensure the region matches your setup
        )
    )
index = pc.Index(index_name)

# Define function to send a message
async def send_message(message, user_message, is_private):
    try:
        # Generate response using handle_response
        response, embedding = handle_response(user_message)

        # Ensure embedding is a float32 numpy array
        embedding = np.array(embedding, dtype=np.float32)

        # Query Pinecone
        query_result = index.query(embedding, top_k=5)

        # Send the query result
        result_text = f"Query Result: {query_result}"
        if is_private:
            await message.author.send(result_text)
        else:
            await message.channel.send(result_text)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the bot
def run_discord_bot():
    TOKEN = os.getenv('TOKEN')

    # Set intents
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

        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)

        print(f"{username} said: '{user_message}' in ({channel})")

        # If message starts with '?', make it private
        if user_message.startswith('?'):
            user_message = user_message[1:]  # Remove '?'
            await send_message(message, user_message, is_private=True)
        else:
            await send_message(message, user_message, is_private=False)

    client.run(TOKEN)
