from flask import Flask, render_template, jsonify, request
from src.utils import *
from pinecone import Pinecone as PineconeClient

from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI



import os

app = Flask(__name__)

load_dotenv()

pc = PineconeClient(
    api_key=os.environ.get("PINECONE_API_KEY")
)
index_name='langchain-medical-chatbot'
index=pc.Index(index_name)
embeddings=download_embeddings()


text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)
logging.info("1")

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
"""llm=ChatOpenAI(openai_api_key='',model_name="gpt-3.5-turbo", temperature=0.5)"""
logging.info("2")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    chain_type_kwargs=chain_type_kwargs
)
logging.info("3")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
