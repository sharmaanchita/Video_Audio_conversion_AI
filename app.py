import streamlit as st
import requests
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from google.oauth2 import service_account
import numpy as np
import math
from pydub import AudioSegment
from io import BytesIO
import tempfile

credentials = service_account.Credentials.from_service_account_file('poc4intern-fb40e8306541.json')

def split_audio(audio, chunk_length_ms=30000):
    chunks = math.ceil(len(audio) / chunk_length_ms)
    audio_chunks = [audio[i * chunk_length_ms: (i + 1) * chunk_length_ms] for i in range(chunks)]
    return audio_chunks

def split_text(text, max_length=3000):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def mono_conversion(audio):
    mono_audio = audio.set_channels(1)
    return mono_audio
def transcription(audio_bytes):
    client = speech.SpeechClient(credentials=credentials)
    
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
    mono_audio = mono_conversion(audio)
    
    audio_chunks = split_audio(mono_audio)
    transcriped = ""
    
    for i, chunk in enumerate(audio_chunks):
        chunk_io = BytesIO()
        chunk.export(chunk_io, format="wav")
        chunk_io.seek(0)
        audio_content = chunk_io.read()

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)

        for result in response.results:
            transcriped += result.alternatives[0].transcript + " "

    return transcriped.strip()

def text_correction(text, azure_openai_key, azure_openai_endpoint):
    headers = { "Content-Type": "application/json", "api-key": azure_openai_key }
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that corrects grammar and removes filler words."},
            {"role": "user", "content": f"Please correct the following text, removing grammatical errors and filler words: {text}"}
        ],
        "max_tokens": 500
    }
    
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Failed to correct text: {response.status_code} - {response.text}")


def text_to_speech(text):
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    text_chunks = split_text(text)
    
    audio_segments = []

    for chunk in text_chunks:
        input_synthesis = texttospeech.SynthesisInput(text=chunk)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Journey-D")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        
        response = client.synthesize_speech(input=input_synthesis, voice=voice, audio_config=audio_config)
        audio_io = BytesIO(response.audio_content)
        audio_segment = AudioSegment.from_file(audio_io, format="wav")
        audio_segments.append(audio_segment)
    
    return audio_segments

# Syncing the new audio to match the original video duration
def sync_audio(audio_segments, target_duration):
    combined_audio = AudioSegment.empty()
    for segment in audio_segments:
        combined_audio += segment

    combined_audio_duration = len(combined_audio) / 1000.0
    if combined_audio_duration < target_duration:
        loops = int(np.ceil(target_duration / combined_audio_duration))
        combined_audio = combined_audio * loops
        combined_audio = combined_audio[:int(target_duration * 1000)]
    
    return combined_audio

def replace_audio(video_bytes, audio_segment):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_temp:
        video_temp.write(video_bytes)
        video_temp.flush()

        video_clip = VideoFileClip(video_temp.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp:
            audio_segment.export(audio_temp.name, format="wav")
            audio_temp.flush()

            audio_clip = AudioFileClip(audio_temp.name)
            video_clip = video_clip.set_audio(audio_clip)

            output_video_io = BytesIO()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_video_temp:
                video_clip.write_videofile(output_video_temp.name, codec="libx264", audio_codec="aac")
                output_video_temp.flush()
                output_video_io.write(output_video_temp.read())

    output_video_io.seek(0)
    return output_video_io

def main():
    azure_openai_key = st.secrets["azure_openai_key"]
    azure_openai_endpoint = st.secrets["azure_openai_endpoint"]

    st.markdown("<h1 style='text-align:center; color:#cc99ff;'>🎬 Video Voice Converter</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#ccccff;'>Swap the audio with Journey voice</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='color:#ccccff;'>Upload your video file:</h5>", unsafe_allow_html=True)

    video_file = st.file_uploader("", type=["mp4"])

    if video_file is not None:
        video_bytes = video_file.read()

        with st.spinner('Processing...'):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video.write(video_bytes)
                temp_video.flush()
                video_clip = VideoFileClip(temp_video.name)
            
            audio_bytes_io = BytesIO()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                video_clip.audio.write_audiofile(temp_audio.name)
                temp_audio.flush()
                audio_bytes_io.write(temp_audio.read())

            audio_bytes_io.seek(0)
            
            # Transcription and text correction
            transcribed_text = transcription(audio_bytes_io.read())
            corrected_text = text_correction(transcribed_text, azure_openai_key, azure_openai_endpoint)
            
            # Generate new audio from corrected text
            new_audio_segments = text_to_speech(corrected_text)

            # Sync new audio to video duration
            adjusted_audio_segment = sync_audio(new_audio_segments, video_clip.duration)

            # Replace original audio with new audio in video
            output_video_io = replace_audio(video_bytes, adjusted_audio_segment)

        st.success("Your video has been successfully processed!")
        st.video(output_video_io)

if __name__ == "__main__":
    main()
