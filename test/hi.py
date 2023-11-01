import os
from PyPDF2 import PdfReader
from bark import SAMPLE_RATE, generate_audio, preload_models
import soundfile as sf
import textwrap
import numpy as np

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

def text_to_speech_with_bark(text, split_into_chunks=True):
    # Preload the models
    preload_models()

    # Split the text into chunks of 250 characters each if split_into_chunks is True
    text_chunks = textwrap.wrap(text, width=250) if split_into_chunks else [text]

    # Initialize an empty list to hold all audio arrays
    all_audio_arrays = []

    # Process each chunk separately and add the results to all_audio_arrays
    for chunk in text_chunks:
        audio_array = generate_audio(chunk , history_prompt="v2/en_speaker_2")
        all_audio_arrays.append(audio_array)

    # Concatenate all audio arrays
    final_audio_array = np.concatenate(all_audio_arrays)

    # Write the final audio to a file
    sf.write("speech.wav", final_audio_array, samplerate=SAMPLE_RATE)

    print("Conversion completed.")

# Path to your file
file_path = '100west.txt'  

# Extract text from file
file_text = extract_text_from_file(file_path)

# Convert extracted text to speech without splitting into chunks
text_to_speech_with_bark(file_text, split_into_chunks=True)