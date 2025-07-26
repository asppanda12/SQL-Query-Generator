# Cleaned and structured version of the RAG pipeline for Streamlit chatbot

import pandas as pd
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
import re

# ---------------------------- CONFIG ----------------------------
WEAVIATE_URL = "<your_weaviate_url>"
WEAVIATE_API_KEY = "<your_weaviate_api_key>"
OPENROUTER_API_KEY = "<your_openrouter_api_key>"
SALES_CSV = "Superstore_sales.csv"
DESC_EXCEL = "Dataset_description.xlsx"

# ---------------------------- DATA LOADING ----------------------------
sales_data = pd.read_csv(SALES_CSV, encoding='ISO-8859-1')
sales_desc = pd.read_excel(DESC_EXCEL, sheet_name='Data_description')
metrics_desc = pd.read_excel(DESC_EXCEL, sheet_name='Metrics_description')

# ---------------------------- EMBEDDING SETUP ----------------------------
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=WEAVIATE_API_KEY)

collections = {
    "SalesData": [
        ("name", DataType.TEXT),
        ("description", DataType.TEXT),
        ("example", DataType.TEXT),
        ("data_type", DataType.TEXT)
    ],
    "MetricData": [
        ("Metric", DataType.TEXT),
        ("Description", DataType.TEXT),
        ("Formula", DataType.TEXT),
        ("Example", DataType.TEXT),
        ("UseCase", DataType.TEXT)
    ]
}

for name, props in collections.items():
    if not client.collections.exists(name):
        client.collections.create(
            name=name,
            properties=[Property(name=p[0], data_type=p[1]) for p in props],
            vector_config=Configure.Vectors.self_provided()
        )

# ---------------------------- QUERY DEFINITION ----------------------------
query = "How have monthly sales and profits varied across different regions over the last two years?"

# ---------------------------- VECTOR SEARCH ----------------------------
def search_collection(name, query_text, limit):
    collection = client.collections.get(name)
    query_vector = model.encode(query_text)
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
        return_metadata=["distance"]
    )
    return response.objects

result_sales_data = search_collection("SalesData", query, 19)
result_metrics_data = search_collection("MetricData", query, 4)

# ---------------------------- FORMAT RESULTS ----------------------------
def build_structured_table(results, fields):
    return [
        {key: getattr(x.properties, key, None) for key in fields}
        for x in results
    ]

sales_data_table = build_structured_table(result_sales_data, ["name", "description", "example", "data_type"])
metrics_data_table = build_structured_table(result_metrics_data, ["metric", "description", "formula", "example", "useCase"])

# ---------------------------- LLM SQL GENERATION ----------------------------
system_prompt = f"""
You are an expert in PySpark SQL query generation.
Generate a valid PySpark SQL query (no explanation) based on:
- Columns: {sales_data_table}
- Metrics: {metrics_data_table}
- Question: {query}
Assume the DataFrame is registered as `sales_data`.
"""

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free", base_url="https://openrouter.ai/api/v1", openai_api_key=OPENROUTER_API_KEY)
response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])

# ---------------------------- SQL CLEANING ----------------------------
def extract_sql(query_str: str) -> str:
    query_str = re.sub(r'<think>.*?</think>', '', query_str, flags=re.DOTALL)
    match = re.search(r"```sql\s*(.*?)\s*```", query_str, re.DOTALL)
    return match.group(1).strip() if match else query_str.strip()

sql_query = extract_sql(response.content)

# ---------------------------- SPARK EXECUTION ----------------------------
spark = SparkSession.builder.appName("SalesRegionQuery").getOrCreate()
df = spark.read.option("header", "true").option("inferSchema", "true").csv(SALES_CSV)
df = df.withColumn("Order Date", to_date("Order Date", "M/d/yyyy"))
df = df.withColumn("Ship Date", to_date("Ship Date", "M/d/yyyy"))
df.createOrReplaceTempView("sales_data")

result_df = spark.sql(sql_query)
pandas_df = result_df.toPandas()

# ---------------------------- VISUALIZATION PROMPT ----------------------------
viz_prompt = f"""
You are a Python visualization expert with deep expertise in matplotlib.
Given the query: {query} and the schema: {pandas_df.columns.tolist()}:
Generate a matplotlib plot. Only return code. No explanations.
"""

llm_viz = ChatOpenAI(model="qwen/qwen3-coder:free", base_url="https://openrouter.ai/api/v1", openai_api_key=OPENROUTER_API_KEY)
response_viz = llm_viz.invoke([SystemMessage(content=viz_prompt), HumanMessage(content="Generate visualization")])

# ---------------------------- EXTRACT CODE ----------------------------
def extract_code(ai_message: AIMessage) -> str:
    content = ai_message.content
    if content.startswith("```") and content.endswith("```"):
        return "\n".join(content.split("\n")[1:-1])
    return content

viz_code = extract_code(response_viz)

# ---------------------------- EXECUTE FINAL CODE ----------------------------
exec(viz_code)
