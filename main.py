from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
import pandas as pd
from langchain_community.document_loaders import SeleniumURLLoader

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize LLM and Embeddings (do this OUTSIDE the Streamlit app flow)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

index_name = "rag00"
pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    index = pc.Index(index_name)
except Exception as e:
    st.warning(f"Index '{index_name}' not found. Creating a new index.")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
retriever = vector_store.as_retriever() # Initialize retriever outside the upload logic

# Streamlit app

st.title("RAG Chatbot")

# Use Streamlit's session state to store whether embeddings have been created
if "embeddings_created" not in st.session_state:
    st.session_state.embeddings_created = False

# User input for data source
option = st.selectbox(
    "How would you like to provide your data?",
    ("Select an option", "CSV", "PDF", "Text file", "URL"),
)

uploaded_data = None
data_name = None

if option != "Select an option":
    if option == "CSV":
        uploaded_data = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_data:
            data_name = "uploaded_csv"
    elif option == "PDF":
        uploaded_data = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_data:
            data_name = "uploaded_pdf"
    elif option == "Text file":
        uploaded_data = st.file_uploader("Choose a Text file", type=["txt"])
        if uploaded_data:
            data_name = "uploaded_text"
    elif option == "URL":
        uploaded_data = st.text_input("Enter URL")
        if uploaded_data:
            data_name = "entered_url"

    if uploaded_data and not st.session_state.embeddings_created: # Only create embeddings if not already done
        try:
            with st.spinner('Processing data and creating embeddings...'):
                if data_name == "uploaded_csv":
                    df = pd.read_csv(uploaded_data)
                    text_data = df.to_string()
                elif data_name == "uploaded_pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_data)
                    text_data = ""
                    for page in pdf_reader.pages:
                        text_data += page.extract_text()
                elif data_name == "uploaded_text":
                    text_data = uploaded_data.getvalue().decode("utf-8")
                elif data_name == "entered_url":
                    loader = SeleniumURLLoader(urls=[uploaded_data])
                    data = loader.load()
                    print(data)
                    if data:
                        text_data = data[0].page_content
                    else:
                        st.error("Could not load URL content.")
                        st.stop()

                documents = [Document(page_content=text_data)]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                docs = text_splitter.split_documents(documents)

                for doc in tqdm(docs):
                    metadata = {"text": f"{doc.page_content}"}
                    vector = embeddings.embed_query(doc.page_content)
                    doc_id = str(hash(doc.page_content))
                    index.upsert(vectors=[(doc_id, vector, metadata)])

                st.session_state.embeddings_created = True  # Set the flag
                st.success("Embeddings created successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    elif uploaded_data is None and option != "Select an option":
        st.warning("Please upload a file or enter a URL.")

# Querying (this now happens INDEPENDENTLY of embedding creation)

if st.session_state.embeddings_created: # Only allow querying if embeddings have been created
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever)
    query = st.text_input("Enter your query")
    if query:
        with st.spinner('Getting response...'):
            response = qa_chain.invoke(query)
        st.write(response)
elif option == "Select an option":
    st.info("Please select an option to proceed.")

elif not st.session_state.embeddings_created and option != "Select an option" and not uploaded_data:
    st.info("Please upload data to start querying.")