import pandas as pd
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
import re
import openpyxl
from Know_base_builder import RAG
import os
from dotenv import load_dotenv
load_dotenv()
class AgenticAi:
    def __init__(self):
        self.rag = RAG()
        self.sales_data = []
        self.metrics_data_table=[]
        self.query=""
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.weaviate_url = os.getenv("WEAVIATE_URL")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.spark = SparkSession.builder \
            .appName("SalesRegionQuery") \
            .getOrCreate()
    def sql_model_creator(self):
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1-0528:free",
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_api_key,
            temperature=0.0
        )
        return llm
    def python_code_creator(self):
        llm = ChatOpenAI(
            model="qwen/qwen3-coder:free",
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_api_key,
            temperature=0.0,
        )
        return llm
    def extract_sql_only(self, ai_message_content: str) -> str:
    # Remove <think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', ai_message_content, flags=re.DOTALL)

        # Extract the SQL block (inside triple backticks or just standalone SQL)
        sql_match = re.search(r"```sql\s*(.*?)\s*```", cleaned, re.DOTALL)

        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # fallback in case the model didn't format with markdown
            sql_match = re.search(r"SELECT\s.*?;", cleaned, re.DOTALL | re.IGNORECASE)
            sql_query = sql_match.group(0).strip() if sql_match else ""

        return sql_query

    def  sql_creator(self,sales_data_table, metrics_data_table, query,llm):
        system_prompt =  f"""
You are an expert in generating **correct and optimized PySpark SQL queries** for business analysts and data teams.

You will be given:
1. A list of **column definitions** for a DataFrame named `sales_data`.
2. A list of **predefined business metrics**, including their descriptions, formulas, and use cases.
3. A **natural language question** asked by a business stakeholder.

Your task:
Translate the question into a valid PySpark SQL query that can run using `spark.sql()` on a registered table named `sales_data`.

---

**Your output should include the following sections**:

### 1. 🧠 Step-by-step Reasoning:
- Break down the question logically.
- Identify key metrics, columns, filters, groupings, date windows, or sorting logic involved.
- Explain how any **business metrics** from the given list are used or derived.
- Explicitly mention how **Spark SQL limitations or syntax** are handled (e.g., date functions, interval support).
- Ensure that any **aggregations** (SUM, COUNT, etc.) and **conditions** (WHERE, HAVING) are correct and well-justified.

### 2. ✅ Final SQL:
- Provide only the final PySpark SQL query.
- It must be **runnable inside `spark.sql("...")`**.
- Avoid invalid syntax not supported in Spark SQL (e.g., `INTERVAL '3 MONTH'`, use `DATE_SUB`, `ADD_MONTHS`, etc.).
- Use `ROUND()` for precision in numerical results when needed.
- Always reference column names as they appear in the schema.

---

**Data References**:
- **Schema** (column definitions from `sales_data` in JSON format):
```json
**List of columns from `sales_data` in json format**:
{sales_data_table}


**List of metrics from `metrics_data` in json format**:
{metrics_data_table}

Example Question:
1. "Show me the top 10 cities by total sales in the last calendar year."
output:
```sql
SELECT 
    City, 
    ROUND(SUM(Sales), 2) AS total_sales
FROM 
    sales_data
WHERE 
    YEAR(`Order Date`) = YEAR(CURRENT_DATE()) - 1
GROUP BY 
    City
ORDER BY 
    total_sales DESC
LIMIT 10
```
2. "Compare the sales performance of corporate vs consumer segments across all regions"
```sql
SELECT 
    Region,
    ROUND(SUM(CASE WHEN Segment = 'Corporate' THEN Sales ELSE 0 END), 2) AS Corporate_Sales,
    ROUND(SUM(CASE WHEN Segment = 'Consumer' THEN Sales ELSE 0 END), 2) AS Consumer_Sales
FROM 
    sales_data
WHERE 
    Segment IN ('Corporate', 'Consumer')
GROUP BY 
    Region
ORDER BY 
    Region
```
3."Which products had a profit margin lower than 5% and were sold more than 50 times?"
```sql
SELECT 
    `Product ID`, 
    `Product Name`
FROM 
    sales_data
GROUP BY 
    `Product ID`, 
    `Product Name`
HAVING 
    SUM(Profit) / SUM(Sales) < 0.05 
    AND COUNT(*) > 50
```
"""

        response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
        print("Response from LLM:", response.content)
        output=self.extract_sql_only(response.content)
        return output
    def spark_execution(self, sql_query):
        df = df = self.spark.read \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .option("multiLine", "true") \
                    .option("quote", "\"") \
                    .option("escape", "\"") \
                    .csv(self.rag.sales_csv)
        df = df.withColumn("Order Date", to_date("Order Date", "M/d/yyyy"))
        df = df.withColumn("Ship Date", to_date("Ship Date", "M/d/yyyy"))
        df.createOrReplaceTempView("sales_data")
        result_df = self.spark.sql(sql_query)
        pandas_df = result_df.toPandas()
        return pandas_df

    def extract_code_from_ai_message(self,ai_message: AIMessage) -> str:
        content = ai_message.content
        if content.startswith("```") and content.endswith("```"):
            return "\n".join(content.split("\n")[1:-1])  # remove the first and last line
        return content  # fallback in case no code block
    def code_extractor(self, response):
        code = self.extract_code_from_ai_message(response)
        return code
    def visualize(self, query, pandas_df, llm_viz):
        list_of_columns = pandas_df.columns.tolist()
        map_visualization_system_prompt = f"""
You are a Python visualization expert with deep expertise in matplotlib.

Given the query: {query} and the schema: {list_of_columns}:
1. Identify the most suitable matplotlib chart type.
2. Use `df` as the variable name for the main DataFrame.
3. Write only the Python matplotlib code to generate the chart.
4. Strictly output only code. No explanations, no reasoning, no comments, no extra text — only code.
"""
        response = llm_viz.invoke([
            SystemMessage(content=map_visualization_system_prompt),
            HumanMessage(content="Generate a matplotlib visualization for the query and the schema")
        ])
        code_extractor = self.extract_code_from_ai_message(response)
        return code_extractor
        
