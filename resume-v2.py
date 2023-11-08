import os
from PyPDF2 import PdfReader
from bark import generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk
import time
from colorama import Fore, Style
import re
import json
from scipy.io.wavfile import read as read_wav

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

def save_chunk_to_file(chunk_audio, index, output_folder):
    chunk_filename = f"{output_folder}/chunk_{index:03d}.wav"
    write_wav(chunk_filename, SAMPLE_RATE, chunk_audio)
    return chunk_filename

def combine_audio_chunks(output_folder, combined_filename):
    chunk_files = [
        os.path.join(output_folder, f) for f in os.listdir(output_folder) 
        if f.endswith('.wav') and re.search(r"chunk_(\d+).wav", f)
    ]
    chunk_files.sort(
        key=lambda x: int(re.search(r"chunk_(\d+).wav", x).group(1))
    )

    combined_audio = []
    for f in chunk_files:
        sr, audio_data = read_wav(f)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Sample rate mismatch in file {f}")
        combined_audio.append(audio_data)
    combined_audio = np.concatenate(combined_audio)
    write_wav(combined_filename, SAMPLE_RATE, combined_audio)

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

# Function to save the last processed chunk index to a state file
def save_state(chunk_index, output_folder):
    state_file_path = os.path.join(output_folder, 'last_processed_chunk.json')
    with open(state_file_path, 'w') as f:
        json.dump({'last_chunk_processed': chunk_index}, f)

# Function to check if there's a previous state and load the last processed chunk index
def load_last_state(output_folder):
    state_file_path = os.path.join(output_folder, 'last_processed_chunk.json')
    if os.path.exists(state_file_path):
        with open(state_file_path, 'r') as f:
            state_data = json.load(f)
            return state_data.get('last_chunk_processed'), True
    return 0, False  # Return 0 and False if no state file exists
# Set up sample rate
SAMPLE_RATE = 22050
HISTORY_PROMPT = "en_speaker_1"

# Extract text from file
file_path = input("Enter the path to the book file: ")
book_title = os.path.basename(file_path).rsplit('.', 1)[0]  # Extract the book's title
# Remove characters that are not allowed in directory names
book_title_safe = re.sub(r'[\\/*?:"<>|]', "", book_title)
long_string = extract_text_from_file(file_path)

# Directory for audio chunks
default_output_folder = f"audio_chunks_{book_title_safe}"

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

# Ask for the directory to save audio chunks, use a unique book name-based folder if none is provided
output_folder = input(f"Enter the directory to save audio chunks (leave blank to use the default '{default_output_folder}' directory): ")
if not output_folder.strip():  # Check if the input is empty or just whitespace
    output_folder = default_output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created default directory based on book title: {output_folder}")
    else:
        print(f"Using existing directory based on book title: {output_folder}")
else:
    os.makedirs(output_folder, exist_ok=True)
    print(f"Using provided directory: {output_folder}")

# Load the last processed chunk or default to the first chunk
# Check if there's a previously saved state and load the last processed chunk index
last_chunk_processed, has_previous_state = load_last_state(output_folder)

if has_previous_state:
    resume_response = input(f"Would you like to resume from the last processed chunk (Chunk {last_chunk_processed})? [Y/n]: ").strip().lower()
    start_chunk = last_chunk_processed if resume_response in ['', 'y'] else None
else:
    start_chunk = None

if start_chunk is None:
    # This prompt is only needed if the user did not choose to resume
    start_chunk = int(input("Enter the chunk number to start from (or 0 to start from the beginning): "))

# Now ensure that end_chunk is set properly
end_chunk = int(input("Enter the chunk number to end at (or -1 to process all): "))
if end_chunk == -1:
    end_chunk = len(chunks)

# Start processing the chunks
for index, prompt in enumerate(chunks[start_chunk:end_chunk], start=start_chunk):
    print(f"{Fore.GREEN}Processing chunk {index}/{end_chunk - 1}: {prompt[:50]}...{Style.RESET_ALL}")

    start_time = time.time()
    audio_array = generate_audio(prompt, history_prompt=HISTORY_PROMPT)
    end_time = time.time()

    save_chunk_to_file(audio_array, index, output_folder)
    save_state(index, output_folder) 
    print(f"{Fore.BLUE}Finished processing chunk {index}/{end_chunk - 1}{Style.RESET_ALL}")

    processing_time = end_time - start_time
    estimated_time_left = processing_time * (end_chunk - index - 1)
    formatted_time = format_time(estimated_time_left)
    print(f"{Fore.YELLOW}Estimated time left: {formatted_time}{Style.RESET_ALL}")

# Combine audio files if requested
if input("Do you want to combine all chunks into one audio file? (y/n): ").strip().lower() == 'y':
    combined_filename = f"{output_folder}/combined_audio.wav"
    print(f"{Fore.CYAN}Combining chunks into file: {combined_filename}{Style.RESET_ALL}")
    combine_audio_chunks(output_folder, combined_filename)
    print(f"{Fore.GREEN}Combined audio saved to: {combined_filename}{Style.RESET_ALL}")
