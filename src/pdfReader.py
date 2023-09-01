import PyPDF2
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import io
import requests
import pdfminer
from pdfminer.high_level import extract_text
import warnings
warnings.filterwarnings("ignore")

from utils.utils import prep_raws, text_splitter

def chat_with_file(file_path, query, embeddings, chain, sep=None, chunk_size=None, chunk_overlap=None, length_function=None):
    
    rw_txt = prep_raws(file_path)
    
    txts = text_splitter(rw_txt, sep=sep, chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function)
    
    vector_db = FAISS.from_texts(txts, embeddings) # in memory

    docs = vector_db.similarity_search(query)

    results = chain(
        {
            "input_documents":docs,
            "question":query
        },
        return_only_outputs=True
    )

    results_ = results['intermediate_steps'][0]

    return results_


def chat_with_file_url(raw_text, query, embeddings, chain, sep=None, chunk_size=None, chunk_overlap=None, length_function=None):
    
    txts = text_splitter(raw_text, sep=sep, chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function)
    
    vector_db = FAISS.from_texts(txts, embeddings) # in memory

    docs = vector_db.similarity_search(query)

    results = chain(
        {
            "input_documents":docs,
            "question":query
        },
        return_only_outputs=True
    )

    results_ = results['intermediate_steps'][0]

    return results_


