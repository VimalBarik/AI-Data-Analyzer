import streamlit as st
import pandas as pd
from llm import LLMAnalyzer
from processor import DataProcessor
import seaborn as sns
import matplotlib.pyplot as plt
from code_executor import execute_python_code
import re
import logging
from rag import build_faiss_index, retrieve_relevant_chunks

def extract_code_block(text):
    code_block_pattern = r"```(?:python)?\s*([\s\S]+?)```"
    match = re.search(code_block_pattern, text)
    if match:
        code = match.group(1).strip()
        explanation = re.sub(code_block_pattern, '', text, count=1).strip()
        return code, explanation
    else:
        return None, text

def set_black_white_pixel_theme():
    st.markdown(
        """
        <style>
        html, body, [class*="css"], * {
            background-color: #fff !important;
            color: #000 !important;
            font-family: 'VT323', 'Press Start 2P', 'monospace', monospace !important;
            font-size: 18px !important;
        }
        .stApp, .stMarkdown, .stTextInput, .stTextArea, .stButton, .stForm, .stProgress, .stAlert, .stSpinner, .stSubheader, .stHeader, .stDataFrame, .stTable, .stText, .stException, .stError, .stSuccess, .stInfo, .st-bb, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
            background-color: #fff !important;
            color: #000 !important;
            font-family: 'VT323', 'Press Start 2P', 'monospace', monospace !important;
        }
        .stTextInput > div > input, .stTextArea > div > textarea {
            background-color: #eee !important;
            color: #000 !important;
            border: 1px solid #ccc !important;
            font-family: 'VT323', 'Press Start 2P', 'monospace', monospace !important;
        }
        .stButton > button {
            background-color: #eee !important;
            color: #000 !important;
            border: 1px solid #ccc !important;
            font-family: 'VT323', 'Press Start 2P', 'monospace', monospace !important;
        }
        .stProgress > div > div {
            background-color: #000 !important;
        }
        .stProgress > div {
            background-color: #ddd !important;
        }
        .stAlert, .stError, .stSuccess, .stInfo {
            background-color: #eee !important;
            color: #000 !important;
            border: 1px solid #ccc !important;
        }
        .stSpinner > div > div, .stSpinner span, .stSpinner, .stProgress, .stProgress span {
            color: #000 !important;
            font-family: 'VT323', 'Press Start 2P', 'monospace', monospace !important;
            font-size: 18px !important;
        }
        .stSubheader, .stHeader, .stTitle {
            color: #000 !important;
            font-family: 'VT323', 'Press Start 2P', 'monospace', monospace !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #000 !important;
            font-family: 'VT323', 'Press Start 2P', 'monospace', monospace !important;
        }
        #MainMenu, footer {visibility: hidden;}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=VT323&family=Press+Start+2P&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
    )

set_black_white_pixel_theme()

def bw(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title(bw("AI Data Analyzer"))
st.markdown(bw("Upload a dataset and chat with your AI data assistant. Powered by Mistral (Ollama)."))

uploaded_file = st.file_uploader("Upload a CSV, Excel, Parquet, or Feather file", type=["csv", "xlsx", "xls", "parquet", "feather"])

FORBIDDEN_PATTERNS = [
    'pd.read_csv', 'pd.read_excel', 'pd.read_parquet', 'pd.read_feather',
    'open(', 'with open', 'os.remove', 'os.rename', 'os.system', 'subprocess',
    'your_data.csv', 'data.csv', 'read_', 'to_csv', 'to_excel', 'to_parquet', 'to_feather'
]

def code_is_safe(code):
    return not any(pattern in code for pattern in FORBIDDEN_PATTERNS)

if uploaded_file:
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    processor = DataProcessor(temp_path)
    df = processor.df
    eda_summary = processor.get_llm_summary()

    if 'rag_index' not in st.session_state or st.session_state.get('rag_file') != uploaded_file.name:
        with st.spinner(bw("Indexing your data for retrieval-augmented analysis...")):
            rag_index, rag_chunks, rag_embedder = build_faiss_index(df)
        st.session_state.rag_index = rag_index
        st.session_state.rag_chunks = rag_chunks
        st.session_state.rag_embedder = rag_embedder
        st.session_state.rag_file = uploaded_file.name
    else:
        rag_index = st.session_state.rag_index
        rag_chunks = st.session_state.rag_chunks
        rag_embedder = st.session_state.rag_embedder

    st.success(bw("File loaded successfully!"))

    if 'llm' not in st.session_state:
        st.session_state.llm = LLMAnalyzer()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'eda_done' not in st.session_state or st.session_state.get('eda_file') != uploaded_file.name:
        df_head = df.head().to_string()
        dtypes = df.dtypes.to_string()
        shape = df.shape
        missing = df.isnull().sum().to_string()
        nunique = df.nunique().to_string()
        columns = ', '.join(list(df.columns))
        with st.spinner(bw("Analyzing your data (initial EDA)...")):
            eda_response = st.session_state.llm.eda_code_and_explanation(
                df_head, dtypes, shape, missing, nunique, columns
            )
        code, eda_response_str = extract_code_block(eda_response)
        initial_eda_result = None
        initial_eda_error = None
        if code and code_is_safe(code):
            initial_eda_result = execute_python_code(code, df)
        elif code:
            initial_eda_error = bw("Initial EDA code was not safe.")
        else:
            initial_eda_error = bw(f"Initial EDA failed: {eda_response if isinstance(eda_response, str) else str(eda_response)}")
        st.session_state.eda_done = True
        st.session_state.eda_file = uploaded_file.name
        st.session_state.eda_response_str = eda_response_str
        st.session_state.initial_eda_result = initial_eda_result
        st.session_state.initial_eda_error = initial_eda_error
        st.session_state.eda_summary = eda_summary
        st.session_state.chat_history = [("assistant", eda_response_str)]
    else:
        eda_response_str = st.session_state.eda_response_str
        initial_eda_result = st.session_state.initial_eda_result
        initial_eda_error = st.session_state.initial_eda_error
        eda_summary = st.session_state.eda_summary

    st.header(bw("Automated EDA Results"))
    st.subheader(bw("Initial EDA"))
    if initial_eda_result:
        st.text(bw(initial_eda_result.get("stdout", "")))
        if initial_eda_result.get("figures"):
            for fig in initial_eda_result["figures"]:
                st.pyplot(fig)
    if initial_eda_error:
        st.error(bw(initial_eda_error))
    st.markdown(bw(eda_response_str))

    auto_eda_mode = st.sidebar.checkbox('Auto-EDA Mode', value=False)

    if auto_eda_mode:
        st.header(bw("Auto-EDA: Autonomous Data Exploration"))
        if st.button("Run Auto-EDA", key="run_auto_eda"):
            st.session_state.auto_eda_steps = []
            st.session_state.auto_eda_stop = False
            auto_eda_steps = []
            prompt = (
                "You are an autonomous data scientist. Explore the DataFrame 'df' step by step. "
                "At each step, generate Python code to learn something new about the data (e.g., summary stats, missing values, distributions, correlations, etc.). "
                "After each result, decide what to do next. Stop when you have a good understanding or if you reach 5 steps. "
                "After each code block, explain what you learned and what you will do next. "
                "If you are done, say 'EDA complete.'"
            )
            eda_context = st.session_state.eda_summary
            chat_history = [f"system: {prompt}", f"eda_summary: {eda_context}"]
            user_message = "Begin autonomous EDA."
            max_steps = 5
            for step in range(max_steps):
                with st.spinner(bw(f"Auto-EDA Step {step+1}...")):
                    chat_result = st.session_state.llm.chat(
                        eda_context,
                        chat_history,
                        user_message
                    )
                code, explanation = extract_code_block(chat_result["message"])
                if code:
                    st.session_state.chat_history.append(("assistant", f"```python\n{code}\n```"))
                    if not code_is_safe(code):
                        st.session_state.chat_history.append(("assistant", bw("Sorry, this code is not allowed.")))
                        break
                    exec_result = execute_python_code(code, df)
                    code_output = exec_result.get("stdout", "")
                    error = exec_result.get("error")
                    result_vars = exec_result.get("result_vars")
                    plot_present = bool(exec_result.get("figures"))
                    if code_output:
                        st.session_state.chat_history.append(("result", code_output))
                    if error:
                        st.session_state.chat_history.append(("result", f"Error: {error}"))
                    if plot_present:
                        st.session_state.chat_history.append(("result", {"figures": exec_result["figures"]}))
                    with st.spinner(bw("Assistant is explaining the result...")):
                        explanation = st.session_state.llm.explain_code_output(
                            user_message, code_output, error, result_vars, plot_present
                        )
                    st.session_state.chat_history.append(("assistant", explanation))
                    chat_history.append(f"assistant: {explanation}")
                    user_message = "What is your next step?"
                    if "EDA complete" in explanation:
                        break
                else:
                    st.session_state.chat_history.append(("assistant", bw(chat_result["message"])))
                    break
        st.write(bw("Auto-EDA will run up to 5 steps or until the LLM says 'EDA complete.' You can switch back to manual mode at any time."))

    if not auto_eda_mode:
        st.header(bw("Chat with your Data Assistant"))
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(bw(f"**You:** {msg}"))
            elif role == "assistant":
                st.markdown(bw(f"**Assistant:**\n{msg}"))
            elif role == "result":
                if isinstance(msg, dict):
                    if "figures" in msg:
                        for fig in msg["figures"]:
                            st.pyplot(fig)
                    else:
                        st.write(bw(str(msg)))
                else:
                    st.write(bw(str(msg)))

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(bw("Type your question or choose an action (e.g., 'predict price'):"), key="chat_input")
            submitted = st.form_submit_button(bw("Send"))
        if submitted and user_input.strip():
            rag_context = retrieve_relevant_chunks(user_input, rag_index, rag_chunks, rag_embedder, top_k=3)
            rag_context_str = "\n".join(rag_context)
            st.session_state.chat_history.append(("user", user_input))

            direct_answer = None
            explanation = None
            if re.search(r"year.*most.*accident", user_input, re.IGNORECASE):
                year_col = None
                for col in df.columns:
                    if re.search(r"year", col, re.IGNORECASE):
                        year_col = col
                        break
                if year_col is not None:
                    try:
                        year_counts = df[year_col].value_counts()
                        most_accidents_year = year_counts.idxmax()
                        most_accidents_count = year_counts.max()
                        direct_answer = f"The year with the most accidents is {most_accidents_year} with {most_accidents_count} accidents."
                        with st.spinner(bw("Assistant is explaining the result...")):
                            explanation = st.session_state.llm.explain_code_output(
                                user_input, direct_answer, None, None, False
                            )
                    except Exception as e:
                        direct_answer = None
            if direct_answer:
                st.session_state.chat_history.append(("assistant", bw(direct_answer)))
                if explanation:
                    st.session_state.chat_history.append(("assistant", bw(explanation)))
            else:
                with st.spinner(bw("Assistant is thinking...")):
                    system_prompt = st.session_state.eda_summary
                    if rag_context_str:
                        system_prompt = f"Relevant data context for this file (if needed):\n{rag_context_str}\n---\n" + system_prompt
                    chat_result = st.session_state.llm.chat(
                        system_prompt,
                        [f"{role}: {msg}" for role, msg in st.session_state.chat_history],
                        user_input
                    )
                code, chat_response_str = extract_code_block(chat_result["message"])
                if code:
                    st.session_state.chat_history.append(("assistant", f"```python\n{code}\n```"))
                    if not code_is_safe(code):
                        st.session_state.chat_history.append(("assistant", bw("Sorry, this code is not allowed. Please try a different file or check your data format.")))
                    else:
                        exec_result = execute_python_code(code, df)
                        code_output = exec_result.get("stdout", "")
                        error = exec_result.get("error")
                        result_vars = exec_result.get("result_vars")
                        plot_present = bool(exec_result.get("figures"))
                        if code_output:
                            st.session_state.chat_history.append(("result", code_output))
                        if error:
                            st.session_state.chat_history.append(("result", f"Error: {error}"))
                        if plot_present:
                            st.session_state.chat_history.append(("result", {"figures": exec_result["figures"]}))
                        with st.spinner(bw("Assistant is explaining the result...")):
                            explanation = st.session_state.llm.explain_code_output(
                                user_input, code_output, error, result_vars, plot_present
                            )
                        st.session_state.chat_history.append(("assistant", explanation))
                else:
                    st.session_state.chat_history.append(("assistant", bw(chat_result["message"])))