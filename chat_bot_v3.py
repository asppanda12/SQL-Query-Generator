import re
import streamlit as st
from Agentic_ai import AgenticAi
import pandas as pd
import matplotlib.pyplot as plt
import io
import traceback
import sys
import tracemalloc
tracemalloc.start()
import numpy as np
import seaborn as sns
import re
def strip_imports(code: str) -> str:
    """Remove import statements to prevent unsafe exec errors."""
    return "\n".join(
        line for line in code.splitlines()
        if not re.match(r"^\s*(import|from)\s+", line)
    )

# Handle different versions of guardrails
try:
    from guardrails import Guard
    from guardrails.schema import Message
    GUARDRAILS_AVAILABLE = True
except ImportError:
    try:
        from guardrails import Guard
        # For newer versions of guardrails, Message might be imported differently
        try:
            from guardrails import Message
        except ImportError:
            # If Message is not available, we'll create a simple replacement
            class Message:
                def __init__(self, role, content):
                    self.role = role
                    self.content = content
        GUARDRAILS_AVAILABLE = True
    except ImportError:
        st.warning("Guardrails not available. Running without validation.")
        GUARDRAILS_AVAILABLE = False
        Guard = None
        Message = None

# Configure Streamlit page
st.set_page_config(
    page_title="Data ChatBot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-indicator {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🤖 Data ChatBot</h1>', unsafe_allow_html=True)
st.markdown("""
Welcome to the Data ChatBot! This tool helps you:
1. **Generate SQL queries** from natural language
2. **Execute queries** and display results
3. **Create visualizations** automatically

Just ask a question about your data and let the AI do the work!
""")

# Initialize the AgenticAi instance
@st.cache_resource
def get_agentic_ai():
    """Initialize and cache the AgenticAi instance"""
    try:
        return AgenticAi()
    except Exception as e:
        st.error(f"Failed to initialize AgenticAi: {e}")
        return None

agentic_ai = get_agentic_ai()

if agentic_ai is None:
    st.stop()

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "step": 1,
        "sql": None,
        "df": None,
        "plot_code": None,
        "user_prompt": None,
        "processing": False,
        "error_log": []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Helper functions
def add_message(role, content):
    """Add a message to the session state"""
    st.session_state.messages.append({"role": role, "content": content})

def log_error(error_msg):
    """Log error to session state for debugging"""
    st.session_state.error_log.append(f"{pd.Timestamp.now()}: {error_msg}")

def validate_sql_with_guardrails(sql_query):
    """Validate SQL query using Guardrails if available"""
    if not GUARDRAILS_AVAILABLE:
        st.info("ℹ️ SQL validation skipped (Guardrails not available).")
        return sql_query
    
    try:
        sql_guard = Guard.from_rail("validate_sql.rail")
        
        # Try different ways to call the guard depending on version
        try:
            # Method 1: Using messages parameter
            validated_output, _ = sql_guard(
                prompt_params={"sql_input": sql_query},
                messages=[Message(role="user", content=sql_query)]
            )
        except TypeError:
            # Method 2: Direct call without messages
            validated_output, _ = sql_guard(
                prompt_params={"sql_input": sql_query}
            )
        
        validated_sql = validated_output.get("validated_sql", sql_query)
        
        if sql_query != validated_sql:
            st.warning("⚠️ Original SQL was modified for safety.")
        
        return validated_sql
        
    except Exception as e:
        st.warning(f"SQL validation failed: {e}. Using original query.")
        log_error(f"SQL validation error: {e}")
        return sql_query

def validate_code_with_guardrails(code_output, user_prompt):
    """Validate Python code using Guardrails if available"""
    if not GUARDRAILS_AVAILABLE:
        st.info("ℹ️ Code validation skipped (Guardrails not available).")
        return code_output
    
    try:
        code_guard = Guard.from_rail("validate_code.rail")
        
        # Try different ways to call the guard depending on version
        try:
            # Method 1: Using messages parameter
            code_validated_output, _ = code_guard(
                prompt_params={"user_prompt": user_prompt},
                messages=[Message(role="user", content=code_output)]
            )
        except TypeError:
            # Method 2: Direct call without messages
            code_validated_output, _ = code_guard(
                prompt_params={"user_prompt": user_prompt}
            )
        
        return code_validated_output.get("plot_code", code_output)
        
    except Exception as e:
        st.warning(f"Code validation failed: {e}. Using original code.")
        log_error(f"Code validation error: {e}")
        return code_output

# Sidebar for controls and status
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    
    # Reset button
    if st.button("🔄 Start Over", type="secondary"):
        # Clear all session state except cached resources
        keys_to_keep = []  # Add any keys you want to keep
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()
    
    # Status indicators
    st.markdown("### 📊 Status")
    status_container = st.container()
    
    with status_container:
        st.write(f"**Current Step:** {st.session_state.step}/3")
        
        # Step indicators
        steps = [
            ("1️⃣ Generate SQL", st.session_state.sql is not None),
            ("2️⃣ Execute Query", st.session_state.df is not None),
            ("3️⃣ Create Chart", st.session_state.plot_code is not None)
        ]
        
        for step_name, completed in steps:
            status = "✅" if completed else "⏳"
            st.write(f"{status} {step_name}")
    
    # Configuration section
    st.markdown("### ⚙️ Configuration")
    show_debug = st.checkbox("Show Debug Info", value=False)
    show_error_log = st.checkbox("Show Error Log", value=False)
    
    # Debug information
    if show_debug:
        st.markdown("### 🐛 Debug Info")
        st.json({
            "Step": st.session_state.step,
            "Has SQL": st.session_state.sql is not None,
            "Has DataFrame": st.session_state.df is not None,
            "DataFrame Shape": st.session_state.df.shape if st.session_state.df is not None else None,
            "Has Plot Code": st.session_state.plot_code is not None,
            "Processing": st.session_state.processing,
            "Guardrails Available": GUARDRAILS_AVAILABLE
        })
    
    # Error log
    if show_error_log and st.session_state.error_log:
        st.markdown("### 🚨 Error Log")
        for error in st.session_state.error_log[-5:]:  # Show last 5 errors
            st.text(error)

# Main content area
st.markdown("### 💬 Chat Interface")

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input
user_prompt = st.chat_input(
    "Ask your query (e.g., 'Show me monthly sales vs profits by region')",
    disabled=st.session_state.processing
)

# Process user input
if user_prompt and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.user_prompt = user_prompt
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    add_message("user", user_prompt)
    
    # ============ STEP 1: Generate SQL ============
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your query and generating SQL..."):
            try:
                # Search for relevant data
                sales_data = agentic_ai.rag.search_sales_data(user_prompt)
                metrics_data = agentic_ai.rag.search_metrics(user_prompt)
                
                # Create LLM and generate SQL
                llm = agentic_ai.sql_model_creator()
                sql_query = agentic_ai.sql_creator(sales_data, metrics_data, user_prompt, llm)
                
                # Validate SQL
                validated_sql = validate_sql_with_guardrails(sql_query)
                st.session_state.sql = validated_sql
                
                # Display results
                st.markdown("### ✅ SQL Query Generated")
                st.code(validated_sql, language="sql")
                
                if sql_query != validated_sql:
                    with st.expander("View Original SQL"):
                        st.code(sql_query, language="sql")
                
                add_message("assistant", f"I've generated the following SQL query:\n```sql\n{validated_sql}\n```")
                st.session_state.step = 2
                
            except Exception as e:
                error_msg = f"Error generating SQL: {str(e)}"
                st.error(error_msg)
                log_error(error_msg)
                add_message("assistant", f"⚠️ {error_msg}")
                
                if show_debug:
                    st.code(traceback.format_exc())
    
    st.session_state.processing = False
    st.rerun()

# ============ STEP 2: Execute SQL and Show DataFrame ============
if (st.session_state.step == 2 and 
    st.session_state.sql and 
    not st.session_state.processing):
    
    if st.button("▶️ Run SQL Query", type="primary"):
        st.session_state.processing = True
        
        with st.chat_message("assistant"):
            with st.spinner("🏃‍♂️ Executing SQL query..."):
                try:
                    df = agentic_ai.spark_execution(st.session_state.sql)
                    
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        
                        # Display DataFrame info
                        st.markdown("### 📋 Query Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", df.shape[0])
                        with col2:
                            st.metric("Columns", df.shape[1])
                        with col3:
                            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                        # Coerce object-type columns to strings or standard types
                        df_fixed = df.copy()
                        for col in df_fixed.columns:
                            if df_fixed[col].dtype == "object":
                                df_fixed[col] = df_fixed[col].astype(str)

                        # Optional: Convert extension dtypes to native types
                        df_fixed = df_fixed.convert_dtypes()
                        # Display DataFrame
                        st.dataframe(df_fixed, use_container_width=True)
                        st.session_state.df = df_fixed
                        # Show data types and basic info
                        with st.expander("📊 Data Info"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Data Types:**")
                                st.write(df_fixed.dtypes.to_frame('Type'))
                            with col2:
                                st.markdown("**Basic Statistics:**")
                                st.write(df_fixed.describe())
                        

                        add_message("assistant", f"Successfully executed the query! Retrieved {df.shape[0]} rows and {df.shape[1]} columns.")
                        st.session_state.step = 3
                        
                    else:
                        st.warning("⚠️ No data returned from the query.")
                        add_message("assistant", "The query executed successfully but returned no data.")
                        
                except Exception as e:
                    error_msg = f"Error executing SQL: {str(e)}"
                    st.error(error_msg)
                    log_error(error_msg)
                    add_message("assistant", f"⚠️ {error_msg}")
                    
                    if show_debug:
                        st.code(traceback.format_exc())
        
        st.session_state.processing = False
        st.rerun()
# ============ Showing the dataframe ============

if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    with st.sidebar.expander("📋 Query Results (Click to expand)", expanded=True):
        st.markdown(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
        st.dataframe(df, use_container_width=True, height=400)
# ============ STEP 3: Generate and Display Visualization ============
if (st.session_state.step == 3 and 
    st.session_state.df is not None and 
    st.session_state.user_prompt and 
    not st.session_state.processing):
    
    if st.button("📊 Generate Visualization", type="primary"):
        st.session_state.processing = True
        
        with st.chat_message("assistant"):
            with st.spinner("🎨 Creating visualization..."):
                try:
                    # Generate visualization code
                    llm_viz = agentic_ai.python_code_creator()
                    code_output = agentic_ai.visualize(
                        st.session_state.user_prompt, 
                        st.session_state.df, 
                        llm_viz
                    )
                    
                    # Validate code
                    safe_plot_code = validate_code_with_guardrails(
                        code_output, 
                        st.session_state.user_prompt
                    )
                    st.session_state.plot_code = safe_plot_code
                    add_message("assistant", "Successfully generated visualization code.")
                    # Display code
                    st.markdown("### 🐍 Generated Python Code")
                    st.code(safe_plot_code, language="python")
                    
                    if code_output != safe_plot_code:
                        with st.expander("View Original Code"):
                            st.code(code_output, language="python")
                    
                    # Execute the visualization code
                    st.markdown("### 📈 Visualization")
                    
                    try:
                        # Create a safe execution environment
                        exec_globals = {
                            "df": st.session_state.df.copy(),  # Use a copy for safety
                            "plt": plt,
                            "io": io,
                            "sns": sns,
                            "np": np,
                            "pd": pd,
                             "__builtins__": {
                                    "range": range,
                                    "len": len,
                                    "min": min,
                                    "max": max,
                                    "sum": sum,
                                    "abs": abs,
                                    "round": round,
                                    "enumerate": enumerate,
                                    "zip": zip,
                                    "list": list,
                                    "dict": dict
                                    # Add more safe functions as needed
                                }
                                                    }
                        safe_plot_code = strip_imports(safe_plot_code)

                        print(safe_plot_code)
                        # Clear any existing plots
                        plt.clf()
                        plt.figure(figsize=(12, 8))
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        exec_globals["fig"] = fig
                        exec_globals["ax1"] = ax1
                        exec_globals["ax2"] = ax2
                        safe_plot_code = safe_plot_code.replace("plt.show()", "").strip()
                        

                        # Execute the plot code
                        exec(safe_plot_code, exec_globals)
                        st.sidebar.pyplot(fig)

                        # # Save and display the plot
                        # buf = io.BytesIO()
                        # plt.savefig(
                        #     buf, 
                        #     format="png", 
                        #     bbox_inches='tight', 
                        #     dpi=150,
                        #     facecolor='white',
                        #     edgecolor='none'
                        # )
                        # buf.seek(0)
                        
                        # st.image(buf, use_column_width=True)
                        
                        # # Provide download option
                        # st.download_button(
                        #     label="📥 Download Chart",
                        #     data=buf.getvalue(),
                        #     file_name=f"chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                        #     mime="image/png"
                        # )
                        
                        # Close the plot to free memory
                        # plt.close('all')
                        
                        add_message("assistant", "Successfully created your visualization! You can download it using the button above.")
                        
                    except Exception as e:
                        error_msg = f"Error executing plot code: {str(e)}"
                        st.error(error_msg)
                        log_error(error_msg)
                        
                        # Show the problematic code for debugging
                        with st.expander("🐛 Debug: Problematic Code"):
                            st.code(safe_plot_code, language="python")
                            st.text(f"Error: {str(e)}")
                            if show_debug:
                                st.code(traceback.format_exc())
                        
                        add_message("assistant", f"⚠️ {error_msg}")
                    
                except Exception as e:
                    error_msg = f"Error creating visualization code: {str(e)}"
                    st.error(error_msg)
                    log_error(error_msg)
                    add_message("assistant", f"⚠️ {error_msg}")
                    
                    if show_debug:
                        st.code(traceback.format_exc())
        
        # Reset to allow new queries
        st.session_state.step = 1
        st.session_state.processing = False
        st.rerun()

# Footer information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 Data ChatBot - Powered by AI | Built with Streamlit</p>
    <p>Ask questions in natural language and get instant SQL queries, data analysis, and visualizations!</p>
</div>
""", unsafe_allow_html=True)

# Add some helpful examples
with st.expander("💡 Example Questions"):
    st.markdown("""
    Here are some example questions you can ask:
    
    **Sales Analysis:**
    - "Show me monthly sales trends for the last year"
    - "What are the top 5 products by revenue?"
    - "Compare sales performance across different regions"
    
    **Customer Analysis:**
    - "Who are our most valuable customers?"
    - "Show customer acquisition trends over time"
    - "Analyze customer demographics by region"
    
    **Performance Metrics:**
    - "Display profit margins by product category"
    - "Show seasonal sales patterns"
    - "Compare this year's performance to last year"
    """)

# Handle any remaining cleanup
if 'processing' in st.session_state:
    if st.session_state.processing:
        st.session_state.processing = False