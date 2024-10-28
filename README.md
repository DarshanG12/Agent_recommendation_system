Agent_Recommendation

This project is a Flask-based application that allows users to query a database of agents and retrieve the top three most relevant agents based on the provided query. The application leverages OpenAI's GPT-3.5-turbo model for conversational retrieval and LangChain for vector storage and retrieval.



The project is organized into the following files:

vectorstore.py: Contains code for generating and loading vector embeddings from agent data stored in an Excel file.
model.py: Defines the conversational retrieval chain using the stored embeddings and a custom prompt template.
app.py: The main Flask application that provides an API endpoint for querying the agent database.


pip install -r requirements.txt
  
You can install the required packages using `pip`:


Environment Setup:
This project uses environment variables for configuration. Create a .env file in the root of your project with the following content:
OPENAI_API_KEY=your_openai_api_key

Running the Flask Application:
I already categorized the agnets  and converted into embeddings as well, you can make use of the same files or else you can create your own categorized and embeddings After all setup start the flask app by running the  below command. 
python app.py
This will start the server, and the API will be accessible at http://127.0.0.1:5000.


Running the Streamlit App:
streamlit run streamlit_app.py

Access the App
Open your browser and navigate to http://localhost:8501 to use the app.
Query the System
Enter your query in the text input field and click the "Get Recommendations" button to receive the top 3 agent recommendations.
