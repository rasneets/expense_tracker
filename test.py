from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the GROQ_API_KEY from your .env file into the environment
load_dotenv()

# Create a connection to a Groq-hosted model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Send a single message and get a response
response = llm.invoke("In one sentence, what is agentic AI?")

print(response.content)