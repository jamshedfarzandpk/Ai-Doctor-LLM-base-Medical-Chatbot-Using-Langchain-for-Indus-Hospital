from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import json
from pinecone import PineconeApiException
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_hugging_face_embeddings()
try:
    index_name = "ai-doctor-llm-medical-chatbot"
    # Embed each chunk and upsert the embeddings into your Pinecone index.
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print(f"✅ Successfully connected to Pinecone index: '{index_name}'")
    # Create a retrieval chain using the vector store and OpenAI model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    # Create the retrieval chain with the defined prompts
    retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    print("✅ Question-answer chain created successfully.")

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ Retrieval chain created successfully.")

    # Flask app setup
    app = Flask(__name__)
    @app.route('/')
    def index():
        return render_template('chat.html')
    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        input = msg
        print(input)
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    if __name__ == '__main__':
        app.run(host="0.0.0.0", port= 8080, debug= True)
except PineconeApiException as e:
    # Handle specific Pinecone API exceptions
    data = json.loads(e.body)
    print(f"❌ Pinecone API Error: {data["message"]}")

except Exception as e:
    # Handle other general exceptions (e.g., network issues)
    print(f"❌ An unexpected error occurred: {e}")