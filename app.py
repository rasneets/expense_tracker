import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))


def get_llm(provider):
    if provider == "Groq (Llama 3.3)":
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0)
    else:
        raise ValueError("Unknown provider")


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

    # --- 3. Automatic insights (plain pandas, no AI, no API calls at all) ---
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

    # --- 4. Model selector ---
    provider = st.selectbox("Choose AI model", ["Groq (Llama 3.3)", "Gemini"])

    # --- 5. Chat interface with history (one API call per question) ---
    st.subheader("Ask a question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("e.g. How much did I spend on Cab?")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        data_as_text = df.to_csv(index=False)
        prompt = f"""You are an assistant answering questions about the user's personal expense data.

Here is the full data in CSV format:
{data_as_text}

Answer the question below using only this data. If you need to sum or compare numbers, work through it carefully and double-check your arithmetic before giving the final number.

Question: {question}"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(provider)
                    response = llm.invoke(prompt)
                    answer = response.content
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        answer = "⚠️ Free API limit reached for now. Please wait a bit and try again."
                    else:
                        answer = f"Something went wrong: {e}"
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload an expense file to get started.")