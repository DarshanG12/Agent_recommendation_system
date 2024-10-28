import pandas as pd
from transformers import pipeline

agents_df = pd.read_excel('Agents.xlsx')

categories = ["Customer Service", "Cleaning", "Programmer", "Q&A Chatbot", "Data Analyst", "Marketing", "Technical Support", "Financial Auditing", "Event Management", "IT Solutions"]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to categorize an agent's description
def categorize_agent(description):
    result = classifier(description, candidate_labels=categories)
    return result['labels'][0]  

# Apply categorization to the agents
agents_df['Category'] = agents_df['Description of Agent'].apply(categorize_agent)

# Save the updated DataFrame to a new Excel file
agents_df.to_excel('Categorized_Agents.xlsx', index=False)
