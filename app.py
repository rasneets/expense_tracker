import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

st.title("💰 My Expense Assistant")

# --- 1. File upload ---
uploaded_file = st.file_uploader("Upload your monthly expense file", type=["xlsx"])

if uploaded_file is not None:

    # --- 2. Load and clean the data ---
    df = pd.read_excel(uploaded_file)
    df = df.iloc[:-1]  # drop totals row (now at the bottom)
    df.columns = df.columns.str.strip()
    category_cols = [c for c in df.columns if c != "Date"]
    df[category_cols] = df[category_cols].fillna(0)

    st.subheader("Your data")
    st.dataframe(df)

    # --- 3. Automatic insights (plain pandas, no AI) ---
    st.subheader("Quick insights")

    totals = df[category_cols].sum()
    top_category = totals.idxmax()
    top_amount = totals.max()
    total_spend = totals.sum()
    avg_daily = total_spend / len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total spend", f"₹{total_spend:,.0f}")
    col2.metric("Top category", f"{top_category}", f"₹{top_amount:,.0f}")
    col3.metric("Avg per day", f"₹{avg_daily:,.0f}")

    # --- 4. Chat box for custom questions ---
    st.subheader("Ask a question")
    question = st.text_input("e.g. How much did I spend on Cab?")

    if question:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            number_of_head_rows=len(df)
        )
        with st.spinner("Thinking..."):
            response = agent.invoke(question)
        st.write(response["output"])

else:
    st.info("Upload an expense file to get started.")