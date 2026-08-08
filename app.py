import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

st.title("My Expense Assistant")

# --- 1. File upload ---
uploaded_file = st.file_uploader("Upload your monthly expense file", type=["xlsx"])

if uploaded_file is not None:

    # --- 2. Load and clean the data ---
    df = pd.read_excel(uploaded_file)
    df = df.iloc[:-1]  # drop totals row (now at the bottom)
    df.columns = df.columns.str.strip()
    category_cols = [c for c in df.columns if c != "Date"]
    df[category_cols] = df[category_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

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

    # --- 4. Chat interface with history ---
    st.subheader("Ask a question")

    # Create the chat history notebook, only once per session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Redraw every past message on the page
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input box (sits at the bottom, like ChatGPT)
    question = st.chat_input("e.g. How much did I spend on Cab?")

    if question:
        # Save and show the user's question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get and show the AI's answer
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            number_of_head_rows=len(df)
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.invoke(question)
            st.write(response["output"])

        # Save the AI's answer too
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})

else:
    st.info("Upload an expense file to get started.")