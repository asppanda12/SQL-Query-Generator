from Agentic_ai import AgenticAi
from Know_base_builder import RAG
from dotenv import load_dotenv  
import os
import pandas as pd
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"


load_dotenv()

agentic_ai = AgenticAi()
query = "How have monthly sales and profits varied across different regions over the last ten years?"
sales_data_table = agentic_ai.rag.search_sales_data(query)
metrics_data_table = agentic_ai.rag.search_metrics(query)
print("Sales Data Table:", sales_data_table)
print("Metrics Data Table:", metrics_data_table)

llm=agentic_ai.sql_model_creator()
sql_creator = agentic_ai.sql_creator(sales_data_table, metrics_data_table, query, llm)
print("Generated SQL Query:", sql_creator)
df= agentic_ai.spark_execution(sql_creator)
print("Result DataFrame:", df)
print(type(df))

list_of_columns=df.columns
llm_viz = agentic_ai.python_code_creator()
code_output=agentic_ai.visualize(query,df,llm_viz)
print(code_output)
exec(code_output)
print("Visualization Code Executed Successfully")
agentic_ai.rag.client.close()
# print("DataFrame:", df.show())
agentic_ai.spark.stop()
