from flask import Flask, request, jsonify
from vectorstore import load_vectorstore
from model import get_conversation_chain

app = Flask(__name__)

# Load the vector store
vectorstore = load_vectorstore()

# Get the conversation chain
conversation_chain = get_conversation_chain(vectorstore)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query')

    if not query_text:
        return jsonify({'error': 'No query provided'}), 400

    # Generate response from the chain
    response = conversation_chain({"question": query_text})

    # Extract the message content
    answer = response.get('answer', 'No answer found')

    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)
