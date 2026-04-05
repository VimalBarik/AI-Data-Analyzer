import pandas as pd
import json
import re
import logging
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import concurrent.futures

class LLMAnalyzer:
    def __init__(self, model_name="mistral", timeout=180):
        self.llm = OllamaLLM(model=model_name)
        self.timeout = timeout

    def _run_with_timeout(self, runnable, input_dict):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: runnable.invoke(input_dict))
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                logging.error("LLM call timed out.")
                return "[ERROR] LLM call timed out."
            except Exception as e:
                logging.error(f"LLM call failed: {e}")
                return f"[ERROR] LLM call failed: {e}"

    def _generate_summary(self, df: pd.DataFrame) -> str:
        summary = {
            "columns": df.columns.tolist(),
            "head": df.head().to_dict(orient="records"),
            "description": df.describe(include='all').to_dict(),
        }
        return json.dumps(summary, indent=2)

    def get_analysis_and_suggestion(self, df: pd.DataFrame) -> dict:
        summary = self._generate_summary(df)
        prompt = PromptTemplate(
            input_variables=["summary"],
            template=(
                "You are a senior data scientist. Based on the following summary of a pandas DataFrame, "
                "provide a brief but insightful analysis of the data and suggest a machine learning task "
                "(either 'classification' or 'regression') that could be performed on it. "
                "Identify a suitable target column for this task and provide a brief justification.\n\n"
                "DataFrame Summary:\n{summary}\n\n"
                "Your output MUST be a JSON object with two keys: 'analysis' (a string) and 'suggestion' "
                "(an object with 'task', 'target', and 'justification' keys)."
            )
        )
        chain = prompt | self.llm
        response = self._run_with_timeout(chain, {"summary": summary})
        try:
            return json.loads(response)
        except Exception:
            return {"error": str(response)}

    def run_suggested_model(self, df: pd.DataFrame, suggestion: dict) -> dict:
        task = suggestion['task'].lower()
        target = suggestion['target']

        if target not in df.columns:
            return {"error": f"Target column '{target}' not found in DataFrame."}

        if task == 'classification':
            n_unique = df[target].nunique()
            if n_unique > 20:
                return {"error": f"Target column '{target}' has {n_unique} unique values, which is too many for classification. Consider using regression or choosing a different target column."}
        if task not in ['classification', 'regression']:
            return {"error": f"Unsupported task: '{task}'"}

        try:
            return {"message": "Please use the chat interface to run ML models. The LLM will generate the appropriate code for your specific request."}
        except Exception as e:
            return {"error": f"An error occurred during model training: {e}"}

    def initial_eda_and_suggestions(self, eda_summary: dict) -> str:
        prompt = PromptTemplate(
            input_variables=["eda_summary"],
            template=(
                "You are a senior data scientist. Here is a detailed EDA summary of a dataset, including column info, unique value counts, correlations, and more.\n"
                "1. Give the user a concise but insightful summary of the data.\n"
                "2. List interesting findings (e.g., strong correlations, high cardinality, missing data, outliers).\n"
                "3. Suggest several possible next actions (e.g., 'predict X', 'cluster Y', 'visualize Z', 'clean missing values'), and for classification, only suggest columns with fewer than 20 unique values as targets.\n"
                "4. Format your response as markdown, with a section for 'Insights' and a section for 'Suggested Actions'.\n"
                "\nEDA Summary (JSON):\n{eda_summary}\n"
            )
        )
        chain = prompt | self.llm
        response = self._run_with_timeout(chain, {"eda_summary": json.dumps(eda_summary)})
        return response

    def chat(self, eda_summary: dict, chat_history: list, user_message: str) -> dict:
        prompt = PromptTemplate(
            input_variables=["eda_summary", "chat_history", "user_message"],
            template=(
                "You are a helpful data science assistant.\n"
                "Here is the EDA summary of the user's data (as JSON):\n{eda_summary}\n"
                "Here is the conversation so far:\n{chat_history}\n"
                "The user says: {user_message}\n"
                "\n"
                "If the user asks for an action (like 'predict X', 'correlation analysis', 'visualize', 'cluster', or any data analysis/modeling), respond in two parts:\n"
                "1. A markdown explanation of what you will do.\n"
                "2. A Python code block (triple backticks, language 'python') that implements the complete solution. Include ALL necessary imports (pandas, numpy, matplotlib, seaborn, scikit-learn, etc.) and write complete, standalone code that works with the DataFrame variable 'df'.\n"
                "If the user just asks a question, only reply in markdown.\n"
                "\n"
                "Example output for a computation request:\n"
                "---\n"
                "I'll create a correlation heatmap for the numeric columns in your dataset.\n"
                "```python\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Get numeric columns\nnumeric_cols = df.select_dtypes(include=[np.number]).columns\ncorr_matrix = df[numeric_cols].corr()\n\n# Create heatmap\nplt.figure(figsize=(10, 8))\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)\nplt.title('Correlation Heatmap')\nplt.tight_layout()\nplt.show()\n\nprint('Correlation matrix:')\nprint(corr_matrix)\n```\n"
                "---\n"
                "Always use the variable 'df' for the DataFrame. Write complete, standalone code with all necessary imports."
            )
        )
        chain = prompt | self.llm
        response = self._run_with_timeout(chain, {
            "eda_summary": json.dumps(eda_summary),
            "chat_history": "\n".join(chat_history),
            "user_message": user_message
        })
        return {"message": response}

    def execute_plan(self, df: pd.DataFrame, plan: dict):
        return {"message": "This feature has been replaced with LLM code generation. Please use the chat interface."}

    def explain_code_output(self, user_question, code_output, error=None, result_vars=None, plot_present=False):
        prompt = PromptTemplate(
            input_variables=["user_question", "code_output", "error", "result_vars", "plot_present"],
            template=(
                "You are a helpful data science assistant.\n"
                "The user asked: {user_question}\n"
                "Here is the output of the code you wrote to answer their question:\n"
                "Output:\n{code_output}\n"
                "Error:\n{error}\n"
                "Result variables:\n{result_vars}\n"
                "Plot present: {plot_present}\n"
                "\n"
                "Please explain the result to the user in clear, concise English. "
                "If there was an error, explain what went wrong and suggest how to fix it. "
                "If a plot was generated (plot_present=True), describe what it shows."
            )
        )
        chain = prompt | self.llm
        response = self._run_with_timeout(chain, {
            "user_question": user_question,
            "code_output": code_output,
            "error": error or "",
            "result_vars": str(result_vars) if result_vars is not None else "",
            "plot_present": str(plot_present)
        })
        return response

    def eda_code_and_explanation(self, df_head, dtypes, shape, missing, nunique, columns):
        prompt = PromptTemplate(
            input_variables=["df_head", "dtypes", "shape", "missing", "nunique", "columns"],
            template=(
                "You are a senior data scientist. The user has uploaded a dataset as a pandas DataFrame called 'df'.\n"
                "The DataFrame 'df' is already loaded and contains the user's data.\n"
                "Never use pd.read_csv or any file loading function. Only use 'df' for all analysis.\n"
                "If you need to reference the data, always use the variable 'df'. Never try to load or read any files.\n"
                "Here are the first few rows:\n{df_head}\n"
                "Column types:\n{dtypes}\n"
                "Shape: {shape}\n"
                "Column names: {columns}\n"
                "Missing value counts:\n{missing}\n"
                "Unique value counts:\n{nunique}\n"
                "\n"
                "Always extract all essential information about the DataFrame (columns, types, shape, missing values, sample rows, etc.) and then perform EDA, no matter what.\n"
                "If the data is unusual or missing headers, infer as much as possible and proceed. Never fail to generate EDA code.\n"
                "If you encounter any errors during EDA, handle and fix them in your code without telling the user about the error. The user should only see EDA insights and suggestions for the next step, never error messages.\n"
                "After the code block, in English, provide insights about the data and suggest what the user can do with it next. Format your explanation as markdown, with a section for 'Insights' and a section for 'Suggested Next Actions'.\n"
                "IMPORTANT: In your markdown explanation, never mention errors, debugging steps, or ask the user to fix anything. Only provide clean insights and suggestions."
            )
        )
        chain = prompt | self.llm
        response = self._run_with_timeout(chain, {
            "df_head": df_head,
            "dtypes": dtypes,
            "shape": str(shape),
            "missing": missing,
            "nunique": nunique,
            "columns": columns
        })
        return response

    def fix_eda_code(self, previous_code, error_message):
        prompt = PromptTemplate(
            input_variables=["previous_code", "error_message"],
            template=(
                "The previous EDA code failed with the following error:\n"
                "{error_message}\n"
                "Please fix the code and try again.\n"
                "Here is the previous code:\n"
                "```python\n{previous_code}\n```\n"
                "Output a new Python code block that fixes the issue and performs EDA as before. Do not show or mention errors to the user."
            )
        )
        chain = prompt | self.llm
        response = self._run_with_timeout(chain, {
            "previous_code": previous_code,
            "error_message": error_message
        })
        match = re.search(r"```python\s*([\s\S]+?)\s*```", response)
        if match:
            return match.group(1)
        return response

FORBIDDEN_PATTERNS = [
    'pd.read_csv', 'pd.read_excel', 'pd.read_parquet', 'pd.read_feather',
    'open(', 'with open', 'os.remove', 'os.rename', 'os.system', 'subprocess',
    'your_data.csv', 'data.csv', 'read_', 'to_csv', 'to_excel', 'to_parquet', 'to_feather'
]

def code_is_safe(code):
    return not any(pattern in code for pattern in FORBIDDEN_PATTERNS)
