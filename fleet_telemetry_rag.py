import os
import requests
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

def fetch_fleet_data():
    """Fetch telemetry data from the fleet API endpoint"""
    api_endpoint = os.getenv("FLEET_API_ENDPOINT")
    bearer_token = os.getenv("FLEET_BEARER_TOKEN")

    if not api_endpoint or not bearer_token:
        return {"error": "API endpoint or bearer token not configured. Check your .env file."}

    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(api_endpoint, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Could not fetch fleet data: {response.status_code} - {response.text}"}

def preprocess_and_store_data():
    """Fetch fleet data and save it to CSV for processing"""
    fleet_data = fetch_fleet_data()
    if "error" in fleet_data:
        print(f"Error fetching data: {fleet_data['error']}")
        return False

    # Handle single dict response
    if isinstance(fleet_data, dict):
        fleet_data = [fleet_data]

    df = pd.DataFrame(fleet_data)
    df.to_csv("fleet_data.csv", index=False)
    print(f"Saved {len(fleet_data)} records to fleet_data.csv")
    return True

def load_and_split_data():
    """Load CSV data and split into chunks for vector database"""
    if not os.path.exists("fleet_data.csv"):
        print("No fleet_data.csv found. Run data preprocessing first.")
        return []

    loader = CSVLoader('fleet_data.csv')
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

def build_vector_database(texts):
    """Create vector database from document chunks"""
    if not texts:
        print("No texts to process")
        return None

    embeddings = HuggingFaceEmbeddings()
    persist_directory = 'docs/chroma_fleet/'

    return Chroma.from_documents(
        documents=texts,
        collection_name="fleet_data",
        embedding=embeddings,
        persist_directory=persist_directory
    )

def build_retrieval_qa_chain(chroma_db):
    """Build the RAG chain for question answering"""
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise ValueError("HuggingFace API token not found. Check your .env file.")

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=10000,
        temperature=0.1,
        huggingfacehub_api_token=hf_token
    )

    retriever = chroma_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # Return top 10 most relevant documents
    )

    template = (
        "You are a Fleet Assistant. Your main role is to help manage and provide insights about the fleet. "
        "If someone asks who or what you are, respond with, "
        "'I am your personal Fleet Assistant'. Do not make any information up. "
        "Only provide factual information. Answer the question below and use the fleet data to provide insights. "
        "If there is no data please let the user know.\n\n"
        "Fleet Data: {context}\n\n"
        "Question: {input}\n\n"
        "Provide a detailed response with any relevant insights."
    )

    prompt = PromptTemplate(input_variables=["context", "input"], template=template)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain

def main():
    """Main application flow"""
    # Check for environment variables
    required_env_vars = ["FLEET_API_ENDPOINT", "FLEET_BEARER_TOKEN", "HUGGINGFACE_API_TOKEN"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all required variables are set.")
        return

    user_query = input("Ask a question about your fleet: ")

    # Fetch and process data
    if not preprocess_and_store_data():
        return

    texts = load_and_split_data()
    if not texts:
        return

    chroma_db = build_vector_database(texts)
    if not chroma_db:
        return

    # Build and run the RAG chain
    chain = build_retrieval_qa_chain(chroma_db)
    response = chain.invoke({"input": user_query})

    print("\nRetrieved Context:")
    print(response['context'])
    print("\nAnswer:")
    print(response['answer'])

if __name__ == "__main__":
    main()