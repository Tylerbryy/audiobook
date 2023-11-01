import os
from PyPDF2 import PdfReader
from bark import generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk


nltk.download('punkt')
preload_models()

def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:  
            text = file.read()
        return text
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf or .txt file.")

# Set up sample rate
SAMPLE_RATE = 22050
HISTORY_PROMPT = "en_speaker_1"

# Path to your file
file_path = 'Animal Farm (George Orwell) (Z-Library).txt'  

# Extract text from file
long_string = extract_text_from_file(file_path)

sentences = nltk.sent_tokenize(long_string)

chunks = ['']
token_counter = 0

for sentence in sentences:
    current_tokens = len(nltk.Text(sentence))
    if token_counter + current_tokens <= 250:
        token_counter = token_counter + current_tokens
        chunks[-1] = chunks[-1] + " " + sentence
    else:
        chunks.append(sentence)
        token_counter = current_tokens

# Generate audio for each prompt
audio_arrays = []
total_chunks = len(chunks)
for index, prompt in enumerate(chunks):
    print(f"Processing chunk {index}: {prompt[:50]}...")  # Print the index and the first 50 characters of the chunk
    print(f"Chunks left to process: {total_chunks - index - 1}")
    audio_array = generate_audio(prompt, history_prompt=HISTORY_PROMPT)
    audio_arrays.append(audio_array)
    print(f"Finished processing chunk {index}")

# Combine the audio files
combined_audio = np.concatenate(audio_arrays)

# Create a filename based on the settings used
filename = f"combined_audio_{HISTORY_PROMPT}_{str(len(chunks))}.wav"

# Write the combined audio to a file
write_wav(filename, SAMPLE_RATE, combined_audio)