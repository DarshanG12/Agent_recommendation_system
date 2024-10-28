from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Prompt templates
template = """
Use the following context (delimited by <ctx></ctx>) to answer the question:
You provide information based on the stored context on the database for answers.
If a question is outside the stored context, you should reply with 'I don't have information on that topic.'
However, it is crucial to remember that you should only answer based on the information stored in documents.
Please suggest the top 3 agents for the user based on the query and follow the below instructions.

Response_generation: Please generate user-friendly responses.
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
Follow-up question: {question}
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
