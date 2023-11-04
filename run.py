import os
from PyPDF2 import PdfReader
from bark import generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk
import time
from colorama import Fore, Style

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
file_path = r'books-txt\Killers of the Flower Moon (David Grann) (Z-Library).txt'  

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
processing_times = []

def format_time(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = ""
    if days > 0:
        time_str += f"{round(days)} days, "
    if hours > 0:
        time_str += f"{round(hours)} hours, "
    if minutes > 0:
        time_str += f"{round(minutes)} minutes and "
    time_str += f"{seconds:.2f} seconds"
    
    return time_str

for index, prompt in enumerate(chunks):
    print(f"{Fore.GREEN}Processing chunk {index}: {prompt[:50]}...{Style.RESET_ALL}")  # Print the index and the first 50 characters of the chunk in green
    
    start_time = time.time()
    audio_array = generate_audio(prompt, history_prompt=HISTORY_PROMPT)
    end_time = time.time()

    processing_time = end_time - start_time
    processing_times.append(processing_time)

    avg_processing_time = sum(processing_times) / len(processing_times)
    remaining_chunks = total_chunks - index - 1
    estimated_time_left = avg_processing_time * remaining_chunks

    minutes, seconds = divmod(estimated_time_left, 60)
        # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    formatted_time = format_time(estimated_time_left)
    print(f"{Fore.YELLOW}Estimated time left: {formatted_time}{Style.RESET_ALL}")

    audio_arrays.append(audio_array)
    print(f"{Fore.BLUE}Finished processing chunk {index}/{total_chunks}{Style.RESET_ALL}")  # Print the finished processing message in blue


# Combine the audio files
combined_audio = np.concatenate(audio_arrays)

# Create a filename based on the settings used
filename = f"combined_audio_{HISTORY_PROMPT}_{str(len(chunks))}.wav"

print(f"{Fore.CYAN}Writing combined audio to file: {filename}{Style.RESET_ALL}")  # Print the filename in cyan

# Write the combined audio to a file
write_wav(filename, SAMPLE_RATE, combined_audio)