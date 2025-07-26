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
import os
from dotenv import load_dotenv
load_dotenv()
class RAG:
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.weaviate_url = os.getenv("WEAVIATE_URL")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.sales_csv = 'Superstore_sales.csv'
        self.desc_excel = 'Dataset_description.xlsx'

        # Load data
        self.sales_data = pd.read_csv(self.sales_csv, encoding='ISO-8859-1')
        self.sales_desc = pd.read_excel(self.desc_excel, sheet_name='Data_description')
        self.metrics_desc = pd.read_excel(self.desc_excel, sheet_name='Metrics_description')

        # Initialize embedding model and Weaviate client
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.client = weaviate.connect_to_weaviate_cloud(cluster_url=self.weaviate_url, auth_credentials=self.weaviate_api_key)
        self.collections = {
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
    def create_collections(self):
        for name, props in self.collections.items():
            if not self.client.collections.exists(name):
                self.client.collections.create(
                    name=name,
                    properties=[Property(name=p[0], data_type=p[1]) for p in props],
                    vector_config=Configure.Vectors.self_provided()
                )
                print(f"✅ Created collection: {name}")
            else:
                print(f"Collection {name} already exists.")
    def fill_sales_data(self):
        for _, row in self.sales_desc.iterrows():
            fullkey = f"{row['Column Name']}_{row['Description']}"
            vector=self.model.encode(fullkey)
            properties = {
                "name": row['Column Name'],
                "description": row['Description'],
                "example": str(row['Example']),
                "data_type":row['Data Type']
            }
            collection = self.client.collections.get("SalesData")
            collection.data.insert(
                properties=properties,
                vector=vector
            )
    def fill_metrics_data(self):
        for _, row in self.metrics_data_desc.iterrows():
            fullkey = f"{row['Metric']}_{row['Description']}_{row['Formula']}"
            vector=self.model.encode(fullkey)
            properties = {
                "Metric": row['Metric'],
                "description": row['Description'],
                "Formula": row['Formula'],
                "example": str(row['Example']),
                "UseCase":row['Use Case']
            }
            collection = self.client.collections.get("MetricData")
            collection.data.insert(
                properties=properties,
                vector=vector
            )
    def search_sales_data(self,query_text, limit=19):
        """Search SalesData collection using vector similarity"""
        # Encode the query text
        query_vector = self.model.encode(query_text)

        # Get the collection
        collection = self.client.collections.get("SalesData")

        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=["distance"]
        )

        return response.objects
    def search_metrics(self, query_text, limit=5):
        """Search MetricsData collection using vector similarity"""
        # Encode the query text
        query_vector = self.model.encode(query_text)

        # Get the collection
        collection = self.client.collections.get("MetricData")

        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=["distance"]
        )

        return response.objects
