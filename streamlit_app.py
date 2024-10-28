import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import pickle
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore(file_path='vectorstores.pkl'):
    with open(file_path, 'rb') as f:
        vectorstore = pickle.load(f)
    return vectorstore

template = """
    Use the following context (delimited by <ctx></ctx>) to answer the question:
    You provide information based on the stored context on the database for answers.
    If a question is outside the stored context, you should reply with 'I don't have information on that topic.
    However, it is crucial to remember that you should only answer based on the information stored in documents.
    Please suggest top 3 agents for user based on the query and follow the below instructions.

    Response_generation:Please generate a user friendly response.
    Agent_name:
    Description:
    Availability:
    Type of agent:
    Category:
    Location:

    {context}
    </ctx>
    ------
    {question}
    Answer:
"""

condense_question_template = """
    Return text in the original language of the follow-up question.
    If the follow-up question does not need context, return the exact same text back.
    Never rephrase the follow-up question given the chat history unless the follow-up question needs context.
    
    Chat History: {chat_history}
    Follow-Up question: {question}
    Standalone question:
"""

prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
condense_question_prompt = PromptTemplate.from_template(condense_question_template)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={'prompt': prompt},
        verbose=True
    )
    return conversation_chain

vectorstore = load_vectorstore()

# Streamlit App
st.title("Agent Recommendation System")

query_text = st.text_input("Enter your query:")
if st.button("Get Recommendations"):
    if not query_text:
        st.error("Please enter a query.")
    else:
        conversation_chain = get_conversation_chain(vectorstore)
        response = conversation_chain({"question": query_text})
        
        # Extract the message content
        answer = response.get('answer', 'No answer found')

        # Display the answer
        st.write("**Response:**")
        st.write(answer)
