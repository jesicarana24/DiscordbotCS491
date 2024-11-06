Capstone FAQ Discord Bot

Capstone FAQ Bot is a tool designed to automate responses to frequently asked questions on a Discord server. Leveraging the power of a Retrieval-Augmented Generation (RAG) model with Pinecone, this bot offers quick and accurate answers based on course and FAQ data, making it a valuable resource for students and TAs alike.

Features

Automated FAQ Responses: Uses a RAG model to generate accurate answers from a structured FAQ database.
Pinecone Integration: Efficient data retrieval with Pinecone as the vector storage solution.
Modular Design: Built to support future integrations with additional platforms.
Feedback Collection: Collects user feedback on responses to continually improve the bot’s performance.
Technologies Used

Bot Framework: Python with Discord.py
Data Retrieval: Pinecone and LangChain for RAG-based retrieval and response generation
Data Storage: CSV files for FAQ data
Project Management: JIRA and GitHub for version control and SCRUM-based workflow management
Getting Started

To set up and run the Capstone FAQ Bot, follow these steps:

Prerequisites
Python 3.x
Pinecone API Access
Discord API Token
Installation
Clone the Repository

git clone https://github.com/jesicarana24/DiscordbotCS491.git
cd DiscordbotCS491
Set Up a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # For Windows: `venv\Scripts\activate`
Install Dependencies

pip install -r requirements.txt
Configure Environment Variables

Create a .env file in the project root directory with the following variables:

DISCORD_TOKEN=your_discord_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
Run the Bot

python bot.py
Usage

Add the Capstone FAQ Bot to a Discord server.
In the designated channel, type a question or keyword related to the course content.
The bot will respond with relevant information pulled from the FAQ data stored in Pinecone.
Contributing

Contributions to the Capstone FAQ Bot are welcome! Please follow these steps:

Fork the repository.
Create a new branch for your feature (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.

