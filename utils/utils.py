import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import requests
import pdfminer
from pdfminer.high_level import extract_text


# prepare the raw tet from file
def prep_raws(file_path):
    doc = PdfReader(file_path)
    raw_text = ''
    for i, page in enumerate(doc.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text


# split the raw text based on some chunks & text overlap
def text_splitter(raw_text, sep=None, chunk_size=None, chunk_overlap=None, length_function=None):
    txt_splitter = CharacterTextSplitter(
        separator=sep,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function
    )
    txts = txt_splitter.split_text(raw_text)

    return txts

# download & convert pdf into text
def downlaod_and_convert_to_text(url):
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the PDF content to a local file
        with open('./data/downloaded.pdf', 'wb') as pdf_file:
            pdf_file.write(response.content)
    else:
        print(f"Failed to download the PDF. Status code: {response.status_code}")

    # Extract text from the downloaded PDF file
    text = extract_text('./data/downloaded.pdf')

    return text