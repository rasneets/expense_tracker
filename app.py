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
        return ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=gemini_api_key, temperature=0)
    else:
        raise ValueError("Unknown provider")


def extract_text(response):
    """Different providers return .content differently — this normalizes it to plain text."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


st.title("💰 My Expense Assistant")

# --- 1. File upload ---
uploaded_file = st.file_uploader("Upload your monthly expense file", type=["xlsx"])
st.caption(
    "Note: expects an .xlsx file with column headers in the first row "
    "(Date + expense categories), one row per day below that, and a "
    "totals row as the very last row."
)

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

Work out the answer carefully and double-check any arithmetic internally, but DO NOT show your reasoning or list out the data you looked at.
Reply with ONLY the final answer, in one short, direct sentence. Do not restate the question.

Question: {question}"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(provider)
                    response = llm.invoke(prompt)
                    answer = extract_text(response)
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        answer = "⚠️ Free API limit reached for now. Please wait a bit and try again."
                    else:
                        answer = f"Something went wrong: {e}"
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload an expense file to get started.")

# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px;">
        <span style="font-size: 12px; color: gray;">
            Developed by Rasneet Singh
            <a href="https://www.linkedin.com/in/rasneet-singh-53476924a?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" style="text-decoration: none; vertical-align: middle;">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="#0A66C2" style="vertical-align: middle; margin-left: 4px;">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
            </a>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)