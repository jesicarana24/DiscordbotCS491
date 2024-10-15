import discord
import os
from dotenv import load_dotenv
from responses import handle_response  # Import handle_response from response.py

# Load environment variables from .env file
load_dotenv()

# Define the function to send a message
async def send_message(message, user_message, is_private):
    try:
        # Generate the response using handle_response
        response = handle_response(user_message)
        
        # Send the response as a private message or to the channel
        if is_private:
            await message.author.send(response)
        else:
            await message.channel.send(response)
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the bot
def run_discord_bot():
    TOKEN = os.getenv('TOKEN')

    # Specify intents
    intents = discord.Intents.default()
    intents.message_content = True  # Ensure message content intent is enabled

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'{client.user} is now running!')

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return  # Ignore the bot's own messages

        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)

        print(f"{username} said: '{user_message}' in ({channel})")

        # Private message if the user starts with '?' 
        if user_message.startswith('?'):
            user_message = user_message[1:]  # Remove the '?' to process the message
            await send_message(message, user_message, is_private=True)
        else:
            await send_message(message, user_message, is_private=False)

    print(TOKEN)
    client.run(TOKEN)
