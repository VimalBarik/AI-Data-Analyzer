import io
import contextlib
import matplotlib.pyplot as plt
import traceback

FORBIDDEN_PATTERNS = [
    'pd.read_csv', 'pd.read_excel', 'pd.read_parquet', 'pd.read_feather',
    'open(', 'with open', 'os.remove', 'os.rename', 'os.system', 'subprocess',
    'your_data.csv', 'data.csv', 'read_', 'to_csv', 'to_excel', 'to_parquet', 'to_feather'
]

def code_is_safe(code):
    return not any(pattern in code for pattern in FORBIDDEN_PATTERNS)

def execute_python_code(code: str, df, extra_globals=None):
    """
    Executes a Python code block with access to 'df' and common data science libraries.
    The LLM should include all necessary imports in the generated code.
    Returns a dict with 'stdout', 'error', 'figures', and optionally 'result_vars'.
    """
    local_vars = {'df': df}
    if extra_globals:
        local_vars.update(extra_globals)
    stdout = io.StringIO()
    result = {}
    # Clear previous matplotlib figures
    plt.close('all')
    try:
        if not code_is_safe(code):
            raise ValueError("Code contains forbidden patterns")
        with contextlib.redirect_stdout(stdout):
            exec(code, globals(), local_vars)
        figs = [plt.figure(n) for n in plt.get_fignums()]
        result['stdout'] = stdout.getvalue()
        result['error'] = None
        result['figures'] = figs
        # Optionally, return variables (e.g., if code sets 'result')
        if 'result' in local_vars:
            result['result_vars'] = local_vars['result']
    except Exception as e:
        result['stdout'] = stdout.getvalue()
        result['error'] = traceback.format_exc()
        result['figures'] = [plt.figure(n) for n in plt.get_fignums()]
    return result 