from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

load_dotenv()

# --- Load and clean the data (same as before) ---
df = pd.read_excel("june.xlsx", sheet_name="july", skiprows=1)
df = df.drop(columns=["Unnamed: 12"], errors="ignore")
df.columns = df.columns.str.strip()
category_cols = [c for c in df.columns if c != "Date"]
df[category_cols] = df[category_cols].fillna(0)

# --- Set up the LLM ---
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# --- Create an agent that can reason over the dataframe ---
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)

# --- Ask it something! ---
question = "How much did I spend on Cab in total?"
response = agent.invoke(question)
print("\nFINAL ANSWER:", response["output"])