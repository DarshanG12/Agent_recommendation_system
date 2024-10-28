import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from dotenv import load_dotenv

load_dotenv()

def create_vectorstore(file_path='Categorized_Agents.xlsx', vectorstore_path='vectorstores.pkl'):
    # Load the categorized agents data
    df = pd.read_excel(file_path)

    # Combine all columns into a single text field
    df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Generate text chunks (optional, here we use the entire text)
    text_chunks = df['combined_text'].tolist()

    # Create vectorstore from text chunks
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # Save vectorstore to a file
    with open(vectorstore_path, 'wb') as f:
        pickle.dump(vectorstore, f)

    print("Embeddings created and saved successfully!")

def load_vectorstore(vectorstore_path='vectorstores.pkl'):
    with open(vectorstore_path, 'rb') as f:
        vectorstore = pickle.load(f)
    return vectorstore
