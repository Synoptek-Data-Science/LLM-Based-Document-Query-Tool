from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import streamlit as st
from streamlit_chat import message

import PyPDF2
from PyPDF2 import PdfReader

import os
from typing_extensions import Concatenate

import io
import requests
import pdfminer
from pdfminer.high_level import extract_text
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from src.pdfReader import chat_with_file, chat_with_file_url
from utils.utils import prep_raws, text_splitter, downlaod_and_convert_to_text


# load open ai key
openai_api_key = os.environ["OPENAI_API_KEY"]

# laod open ai embeddings & model
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="map_rerank", return_intermediate_steps=True)


# Display the page title and the text box for the user to ask the question
st.title("✨ PDF Query Tool")

# Display the page title and the text box for the user to ask the question

# pdf query tool via uploading document
if st.checkbox("Query the uploaded document", key='2'):
    
    # Display the page title and the text box for the user to ask the question
    file_path = st.file_uploader("Upload the file here")
    if file_path:
        user_query = st.text_input("What you want to search in this PDF")
        if user_query:
            resp = chat_with_file(file_path, user_query, embeddings, chain, sep="\n", chunk_size=1000, chunk_overlap=100, length_function=len)

            answer = resp['answer']
            conf_score = resp['score']

            st.write("Answer")
            st.write(answer)
            st.write('\n')
            st.write("Confidence Score")
            st.write(conf_score)

# pdf query tool by url
if st.checkbox("Query the document from url", key='3'):
    url_ = st.text_input("Enter URL here")
    if url_:
        raw_text = downlaod_and_convert_to_text(url_)
        user_query = st.text_input("What you want to search in this PDF")
        if user_query:
            resp = chat_with_file_url(raw_text, user_query, embeddings, chain, sep="\n", chunk_size=1000, chunk_overlap=100, length_function=len)

            answer = resp['answer']
            conf_score = resp['score']

            st.write("Answer")
            st.write(answer)
            st.write('\n')
            st.write("Confidence Score")
            st.write(conf_score)


    

