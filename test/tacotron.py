import os
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
from PyPDF2 import PdfReader
import torch
import textwrap
from tqdm import tqdm
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import torchaudio
import torch
import textwrap
from tqdm import tqdm

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



def text_to_speech_with_tortoise(text, voice='tom', preset='ultra_fast', split_into_chunks=True):
    # Check if a GPU is available and if not, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the TextToSpeech model and move it to the device
    tts = TextToSpeech(use_deepspeed=True, kv_cache=True).to(device)

    # Load the voice samples and conditioning latents
    voice_samples, conditioning_latents = load_voice(voice)
    voice_samples = voice_samples.to(device)
    conditioning_latents = conditioning_latents.to(device) if conditioning_latents is not None else None

    # Split the text into chunks of 250 characters each if split_into_chunks is True
    text_chunks = textwrap.wrap(text, width=250) if split_into_chunks else [text]

    # Initialize an empty tensor to hold the speech
    speech = torch.tensor([]).to(device)

    # Process each chunk separately and concatenate the results
    for chunk in tqdm(text_chunks, desc="Processing text chunks"):
        # Convert the text to speech
        gen = tts.tts_with_preset(chunk, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)

        # Concatenate the generated speech
        speech = torch.cat((speech, gen.squeeze(0)), dim=0)

    # Save the generated speech
    torchaudio.save('generated.wav', speech.cpu(), 24000)

# Path to your file
file_path = '100west.txt'  

# Extract text from file
file_text = extract_text_from_file(file_path)

# Convert extracted text to speech
text_to_speech_with_tortoise(file_text)