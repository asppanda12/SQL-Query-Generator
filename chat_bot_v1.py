import streamlit as st
from Agentic_ai import AgenticAi
from Know_base_builder import RAG
from dotenv import load_dotenv
import os
import pandas as pd
import time

# Setup
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
load_dotenv()

# Initialize
agentic_ai = AgenticAi()
st.title("DATA ANALYTICS AGENT")

# State management
if "step" not in st.session_state:
    st.session_state.step = 1
if "query" not in st.session_state:
    st.session_state.query = ""
if "sql_query" not in st.session_state:
    st.session_state.sql_query = ""
if "df" not in st.session_state:
    st.session_state.df = None

# Step 1: Enter query
def step1():
    query = st.text_input("Enter your analytical query:", value=st.session_state.query)
    if st.button("Generate SQL"):
        st.session_state.query = query
        sales_data_table = agentic_ai.rag.search_sales_data(query)
        metrics_data_table = agentic_ai.rag.search_metrics(query)
        llm = agentic_ai.sql_model_creator()
        sql_creator = agentic_ai.sql_creator(sales_data_table, metrics_data_table, query, llm)
        st.session_state.sql_query = sql_creator
        st.session_state.step = 2

# Step 2: Show SQL and options

def step2():
    st.code(st.session_state.sql_query, language="sql")
    col1, col2 = st.columns(2)
    if col1.button("Generate DataFrame"):
        st.session_state.step = 3
    if col2.button("Back to New Query"):
        st.session_state.step = 1

# Step 3: Generate DataFrame and show options

def step3():
    with st.spinner("Running SQL and fetching results..."):
        df = agentic_ai.spark_execution(st.session_state.sql_query)
        st.session_state.df = df
        st.dataframe(df)

    col1, col2 = st.columns(2)
    if col1.button("Generate Visualization"):
        st.session_state.step = 4
    if col2.button("New Query"):
        st.session_state.step = 1

# Step 4: Visualization and reset to step 1

def step4():
    df = st.session_state.df
    list_of_columns = df.columns
    llm_viz = agentic_ai.python_code_creator()
    code_output = agentic_ai.visualize(st.session_state.query, df, llm_viz)
    st.code(code_output, language="python")

    try:
        exec_globals = {"df": df}
        exec(code_output, exec_globals)
        st.pyplot()
        st.success("Visualization generated successfully")
    except Exception as e:
        st.error(f"Error in executing visualization code: {e}")

    if st.button("Start New Query"):
        st.session_state.step = 1

# Main Flow
if st.session_state.step == 1:
    step1()
elif st.session_state.step == 2:
    step2()
elif st.session_state.step == 3:
    step3()
elif st.session_state.step == 4:
    step4()

# Cleanup
agentic_ai.rag.client.close()
agentic_ai.spark.stop()
