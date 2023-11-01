import os
from PyPDF2 import PdfReader
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from tqdm import tqdm
import textwrap
# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf or .txt file.")
    

def text_to_speech_with_speecht5(text, split_into_chunks=True):
    # Check if using GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Initialize the processor, model, and vocoder
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    # Split the text into chunks of 250 characters each if split_into_chunks is True
    text_chunks = textwrap.wrap(text, width=250) if split_into_chunks else [text]

    # Initialize an empty tensor to hold the speech
    speech = torch.tensor([]).to(device)

    # Process each chunk separately and concatenate the results
    for chunk in tqdm(text_chunks, desc="Processing text chunks"):  # Add progress bar here
        inputs = processor(text=chunk, return_tensors="pt").to(device)
        chunk_speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        if chunk_speech is not None:
            speech = torch.cat((speech, chunk_speech), dim=0)

    # Write the speech audio to a file
    sf.write("speech1.wav", speech.cpu().numpy(), samplerate=16000)

    print("Conversion completed.")

# Path to your file
file_path = '100west.txt'  

# Extract text from file
file_text = extract_text_from_file(file_path)

# Convert extracted text to speech without splitting into chunks
text_to_speech_with_speecht5(file_text, split_into_chunks=True)