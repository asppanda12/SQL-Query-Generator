import streamlit as st
from Agentic_ai import AgenticAi
import pandas as pd
import matplotlib.pyplot as plt
import io
from guardrails import Guard
from guardrails.schema import Message

st.set_page_config(page_title="Data ChatBot", layout="wide")

# Initialize agent
agentic_ai = AgenticAi()

# Setup session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "step" not in st.session_state:
    st.session_state.step = 1  # Step 1: Ask query

if "sql" not in st.session_state:
    st.session_state.sql = None

if "df" not in st.session_state:
    st.session_state.df = None

if "plot_code" not in st.session_state:
    st.session_state.plot_code = None

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Main input from user
user_prompt = st.chat_input("Ask your query (e.g. Monthly sales vs profits by region)?")

if user_prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # ============ STEP 1: Generate SQL ============
    with st.chat_message("assistant"):
        st.markdown("Generating SQL query...")

        sales_data = agentic_ai.rag.search_sales_data(user_prompt)
        metrics_data = agentic_ai.rag.search_metrics(user_prompt)
        llm = agentic_ai.sql_model_creator()
        sql_query = agentic_ai.sql_creator(sales_data, metrics_data, user_prompt, llm)
        sql_guard = Guard.from_rail("validate_sql.rail")
        validated_output, _ = sql_guard(
                            prompt_params={"sql_input": sql_query},
                            messages=[Message(role="user", content=sql_query)]
                        )
        validated_sql = validated_output.get("validated_sql", sql_query)

        st.session_state.sql = validated_sql        
        st.markdown("### ✅ Validated SQL Query")
        if sql_query != validated_sql:
            st.warning("⚠️ Original SQL was modified for safety.")


        st.markdown("### Generated SQL")
        st.code(validated_sql, language="sql")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Here's the generated SQL:\n```sql\n{sql_query}\n```"
        })

    st.session_state.step = 2

# ============ STEP 2: Show DataFrame ============
if st.session_state.step == 2 and st.button("Run SQL and Show DataFrame"):
    with st.chat_message("assistant"):
        st.markdown("Running SQL and fetching DataFrame...")

        df = agentic_ai.spark_execution(st.session_state.sql)
        st.dataframe(df)
        st.session_state.df = df

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Here’s the resulting DataFrame with shape `{df.shape}`."
        })

    st.session_state.step = 3

# ============ STEP 3: Show Chart ============
if st.session_state.step == 3 and st.button("Generate Chart"):
    with st.chat_message("assistant"):
        st.markdown("Creating visualization...")

        llm_viz = agentic_ai.python_code_creator()
        code_output = agentic_ai.visualize(user_prompt, st.session_state.df, llm_viz)
        st.session_state.plot_code = code_output
        code_guard = Guard.from_rail("validate_code.rail")
        code_validated_output, _ = code_guard(prompt_params={"user_prompt": user_prompt})
        safe_plot_code = code_validated_output.get("plot_code", code_output)
        st.session_state.plot_code = safe_plot_code

        st.markdown("### ✅ Visualization Code (Validated)")
        st.code(safe_plot_code, language="python")


        try:
            exec_globals = {"df": st.session_state.df, "plt": plt, "io": io}
            exec(safe_plot_code, exec_globals)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            st.image(buf)
            plt.close()

            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here's the visualization based on your query."
            })

        except Exception as e:
            st.error(f"Error running plot code: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error generating visualization: `{str(e)}`"
            })

    st.session_state.step = 1  # Reset to allow new query

# Optional: Reset chat
with st.sidebar:
    if st.button("🔄 Start Over"):
        st.session_state.clear()
        st.experimental_rerun()
