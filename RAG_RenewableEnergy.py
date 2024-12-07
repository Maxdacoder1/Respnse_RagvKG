from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key not found. Check your .env file.")

# Initialize OpenAI chat model
chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4")

# Load documents from Wikipedia
print("Loading Wikipedia documents...")
raw_documents = WikipediaLoader(query="Renewable Energy").load()

# Split documents into smaller chunks
print("Splitting documents into manageable chunks...")
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents[:3])

# Print the first chunk for verification
print(f"First document chunk: {documents[0].page_content[:200]}...\n")

# Create vector index for RAG
print("Creating vector index for retrieval...")
vector_index = Neo4jVector.from_documents(
    documents,
    OpenAIEmbeddings()
)

# Define the retrieval function
def retriever(question: str):
    print(f"Processing question: {question}")
    # Retrieve unstructured data using vector index
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    if not unstructured_data:
        return "No relevant data found."
    
    # Combine results into a single response
    final_data = f"Retrieved Data:\n{' '.join(unstructured_data)}"
    print(f"Retrieved Data:\n{final_data[:500]}...\n")  # Print a preview of the result
    return final_data

# Test the RAG system with a question
question = "What is Renewable Energy?"
print("Testing RAG system...")
answer = retriever(question)


# Use OpenAI model to generate a concise response
response = chat.invoke(
    f"Based on the following context, answer the question:\n\nContext: {answer}\n\nQuestion: {question}\nAnswer:"
)
print("\nFinal Answer:", response)
