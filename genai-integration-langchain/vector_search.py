import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

from langchain_openai import OpenAIEmbeddings

# Create the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

from langchain_neo4j import Neo4jVector

# Create Vector
plot_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

# Search for similar movie plots
print("Enter your plot here:")
plot = input()
result = plot_vector.similarity_search(plot, k=3)
print(result)

# Parse the documents
for doc in result:
    print(f"Title: {doc.metadata['title']}")
    print(f"Plot: {doc.page_content}\n")
