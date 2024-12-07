from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from langchain_community.cache import InMemoryCache
import langchain

load_dotenv()

# Load environment variables
AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

# Set up caching to avoid redundant LLM calls
langchain.llm_cache = InMemoryCache()

# Initialize ChatOpenAI models
# Use gpt-3.5-turbo for faster intermediate processing
chat_fast = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo")
# Use gpt-4 for the final answer if needed
chat_final = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4")

# Initialize the Neo4j graph database connection
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# Clear existing data in the database
kg.query("MATCH (n) DETACH DELETE n")

# Load Wikipedia pages for the topic using a single WikipediaLoader
queries = ["Mental Health"]
raw_documents = []

# Custom function to load multiple queries using one WikipediaLoader
def load_wikipedia_documents(queries):
    loader = WikipediaLoader(query="")
    documents = []
    for query in queries:
        loader.query = query
        documents.extend(loader.load())
    return documents

# Load documents
raw_documents = load_wikipedia_documents(queries)

# Define chunking strategy with reduced chunk size
text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

# Limit the number of documents to process for faster testing
documents = documents[:5]  # Limited to 5 documents for faster processing

# Initialize LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=chat_fast)

# Process graph transformation synchronously (since dataset is small)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Store transformed documents to Neo4j
res = kg.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)

# Create a vector index with the updated data for hybrid retrieval
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)

# Define a Pydantic model for entity extraction
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

# Simplify the prompt for entity extraction
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract person and organization names from the text.",
        ),
        (
            "human",
            "Text: {question}",
        ),
    ]
)
# Use the faster LLM model for entity extraction
entity_chain = prompt | chat_fast.with_structured_output(Entities)

# Ensure proper indexing in Neo4j for efficient queries
kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    if words:
        full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Define the structured retriever function
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question.
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        print(f"Getting Entity: {entity}")
        response = kg.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el["output"] for el in response]) + "\n"
    return result

# Define the final retrieval function
def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
    """
    print(f"\nFinal Data::: ==>{final_data}")
    return final_data

# Define the RAG chain components
# Prompt to condense a follow-up question
_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Define the search query runnable
_search_query = RunnableBranch(
    # If input includes chat_history, condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | chat_fast
        | StrOutputParser(),
    ),
    # Else, pass through the question as is
    RunnableLambda(lambda x: x["question"]),
)

# Define the final answer prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Build the RAG chain
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | chat_final  # Use GPT-4 for generating the final answer
    | StrOutputParser()
)

# Test the chain with a simple question
res_simple = chain.invoke(
    {
        "question": "What is mental health?",
    }
)

print(f"\nResults === {res_simple}\n\n")

# Verify that the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Check your .env file.")

print(f"Loaded API Key: {api_key[:5]}...")

# Test the chain with a question that includes chat history
res_hist = chain.invoke(
    {
        "question": "What is mental health?",
        "chat_history": [
            (
               "Mental health relates to emotional and psychological well-being.",
               "it encompasses an individual's emotional, psychological, and social well-being, influencing how they think, feel, and behave in daily life."
            )
            
        ],
    }
)

print(f"\nResults === {res_hist}\n\n")
