import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
import traceback
import io
import sys
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Data Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# ------------------ Session State ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

if "groq_client" not in st.session_state:
    st.session_state.groq_client = None

# ------------------ Groq Config ------------------
def configure_groq(api_key):
    try:
        client = Groq(api_key=api_key)
        st.session_state.groq_client = client
        return True
    except Exception as e:
        st.error(f"Failed to configure Groq: {str(e)}")
        return False

# ------------------ Prompt Classifier ------------------
def classify_prompt(prompt, df_info):
    try:
        if not st.session_state.groq_client:
            st.error("Groq client not configured")
            return 1

        today = datetime.today().strftime('%Y-%m-%d')

        classification_prompt = f"""
        Today's date is: {today}
        You are a prompt classifier for a data analysis chatbot.
        Dataset info: {df_info}
        User prompt: "{prompt}"
        Classify this prompt:
        - Return 1 if it's about querying, filtering, summarizing, or analyzing data
        - Return 0 if it's about creating charts, plots, or visualizations
        Examples:
        - "Show me the first 10 rows" → 1
        - "What's the average sales?" → 1
        - "Filter data where age > 30" → 1
        - "Create a bar chart of sales by region" → 0
        - "Plot the correlation between variables" → 0
        - "Show distribution of ages" → 0
        Respond with only the number (0 or 1).
        """

        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": classification_prompt}],
            model="llama3-70b-8192",
            temperature=0,
        )

        return int(chat_completion.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return 1

# ------------------ Code Generator ------------------
def generate_code(prompt, df_info, classification):
    try:
        if not st.session_state.groq_client:
            st.error("Groq client not configured")
            return None

        today = datetime.today().strftime('%Y-%m-%d')

        if classification == 1:
            code_prompt = f"""
            whenever required use datetime.now()

            While building a query use these column names "Serial No.","Lot ID","Manufacturing Date","Expiry Date","Labels Readable"
            when slicing the original dataframe use .loc function 
            If you are filtering the dataframe add .copy() function
            If time handling required in the code then use `import from datetime import datetime, timedelta`

            
            

            You are a Pandas code generator for data analysis.
            Dataset info: {df_info}
            User request: "{prompt}"
            Generate Python code using Pandas to answer this request.
            Rules:
            -Only generate code based on the following user request.
            -Do not reuse previous instructions.
            - If asked about shelf life/distribution it is expiry date - manufacturing date. It should be in number of days
            - Use the variable 'df' for the DataFrame (already loaded)
            - DO NOT modify the original df
            - Store the result in a variable called 'result'
            - Only use Pandas operations
            - No imports needed (pandas already imported as pd)
            - Keep code concise and efficient
            """
        else:
            code_prompt = f"""
            
            While building a query use these column names "Serial No.","Lot ID","Manufacturing Date","Expiry Date","Labels Readable"
            when slicing the original dataframe use .loc function 
            If you are filtering the dataframe add .copy() function
            If time handling required in the code then use `import from datetime import datetime, timedelta`
             whenever required use datetime.now()
            
            You are a Python data visualization assistant using Plotly Express (px).
            when ask about shelf distribution put number of days on y-axis and each lot id as in one bar. 
            whenever asked question include lot id on x-axis treat each lot id as one bar.

            Dataset Description:
            - Each row represents a product unit.
            - `Lot ID` groups products into batches.
            - `Manufacturing Date` and `Expiry Date` are in day-month-year format.
            - `Labels Readable` is a binary field (1 = readable, 0 = not readable).
            - You may compute shelf life as: `(Expiry Date - Manufacturing Date).days`
            - Dates should be parsed as datetime where needed.
            - Serial numbers repeat per lot — not globally unique.
            - "Products expiring in the next 30 days" means:
             

            User Request: "{prompt}"

            Generate Python code using Plotly Express to visualize the answer.
            - Use DataFrame 'df' (already loaded)
            - Create figure object named `fig`
            - Only use Plotly Express (imported as px)
            - Add labels, titles, and color where relevant
            - Keep code compact and readable
            """


        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": code_prompt}],
            model="llama3-70b-8192",
            temperature=0,
        )

        code = chat_completion.choices[0].message.content.strip()

        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code
    except Exception as e:
        st.error(f"Code generation error: {str(e)}")
        return None

# ------------------ Code Executor ------------------
def execute_code(code, df):
    try:
        exec_globals = {
            'df': df,
            'pd': pd,
            'px': px,
            'result': None,
            'fig': None,
            'datetime': datetime
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)

        if stderr_capture.getvalue():
            st.error(f"Execution error: {stderr_capture.getvalue()}")
            return None, None

        return exec_globals.get('result'), exec_globals.get('fig')

    except Exception:
        st.error(f"Code execution failed:\n```\n{traceback.format_exc()}\n```")
        return None, None

# ------------------ DF Info ------------------
def get_dataframe_info(df):
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'sample': df.head(3).to_dict()
    }
    return str(info)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        help="Get your API key from Groq Console"
    )
    if api_key != st.session_state.groq_api_key:
        st.session_state.groq_api_key = api_key
        if api_key:
            configure_groq(api_key)

    st.divider()
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Loaded: {df.shape} rows, {df.shape} columns")
            with st.expander("Dataset Preview"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# ------------------ Main App ------------------
st.title("🤖 Data Chat Assistant")
st.markdown("*Powered by Groq (Llama3 70B)*")

if st.session_state.df is not None and st.session_state.groq_api_key:
    df = st.session_state.df

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                if "result" in message and message["result"] is not None:
                    st.write("**Result:**")
                    st.dataframe(message["result"])
                if "figure" in message and message["figure"] is not None:
                    st.plotly_chart(message["figure"], use_container_width=True, key=str(uuid.uuid4()))
                if "code" in message:
                    st.write("**Generated Code:**")
                    st.code(message["code"], language="python")
                if "error" in message:
                    st.error(message["error"])

    if prompt := st.chat_input("Ask me anything about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                df_info = get_dataframe_info(df)
                classification = classify_prompt(prompt, df_info)
                st.write(f"**Classification:** {'Data Query' if classification == 1 else 'Visualization'}")

                code = generate_code(prompt, df_info, classification)

                if code:
                    st.write("**Generated Code:**")
                    st.code(code, language="python")
                    result, fig = execute_code(code, df)

                    response_data = {"role": "assistant", "code": code}
                    if classification == 1 and result is not None:
                        st.write("**Result:**")
                        st.dataframe(result)
                        response_data["result"] = result
                    elif classification == 0 and fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
                        response_data["figure"] = fig
                    else:
                        error_msg = "No result generated. Please try rephrasing your request."
                        st.error(error_msg)
                        response_data["error"] = error_msg

                    st.session_state.messages.append(response_data)
                else:
                    error_msg = "Failed to generate code. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "error": error_msg
                    })

else:
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        ## Welcome! 👋

        To get started:
        1. **🔑 Add your Groq API Key** in the sidebar  
        2. **📁 Upload a CSV or Excel file**  
        3. **💬 Start chatting** about your data!

        ### What can I do?

        **📊 Data Queries:**
        - "Show me the first 10 rows"
        - "What's the average sales by region?"
        - "Filter customers with age > 30"

        **📈 Visualizations:**
        - "Create a bar chart of sales by month"
        - "Show the distribution of ages"
        - "Plot correlation between variables"

        ---
        *Get your Groq API key from [Groq Console](https://console.groq.com/keys)*
        """)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "Built with Streamlit ❤️ Powered by Groq (Llama3 70B)"
    "</div>",
    unsafe_allow_html=True
)
