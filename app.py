import streamlit as st
import requests
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from google.oauth2 import service_account
import numpy as np
import math
from pydub import AudioSegment 

credentials = service_account.Credentials.from_service_account_file('poc4intern-fb40e8306541.json')

#Splitting to handle HTTP request payload limit - for audio & video both 
def split_audio(file_path, chunk_length_ms=30000): 
    audio = AudioSegment.from_file(file_path)
    chunks = math.ceil(len(audio) / chunk_length_ms)
    audio_chunks = [audio[i * chunk_length_ms: (i + 1) * chunk_length_ms] for i in range(chunks)]
    return audio_chunks


def split_text(text, max_length=3000):  
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


#To handle videos with 2 or more audio channels.
def mono_conversion(input_audio, output_audio):
    audio = AudioSegment.from_file(input_audio)
    mono_audio = audio.set_channels(1)  
    mono_audio.export(output_audio, format="wav")


def transcription(audio):
    client = speech.SpeechClient(credentials=credentials)
    
    audio_mono = "mono_audio.wav"
    mono_conversion(audio, audio_mono)
    
    audio_chunks = split_audio(audio_mono)
    transcriped = ""

    for i, chunk in enumerate(audio_chunks):
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        with open(chunk_path, "rb") as audio:
            content = audio.read()

        audio = speech.RecognitionAudio(content=content)
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
    headers = { "Content-Type": "application/json",
                "api-key": azure_openai_key}
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that corrects grammar and removes filler words."},
            {"role": "user", "content": f"Please correct the following text, removing grammatical errors and filler words: {text}"}],
        "max_tokens": 500}
    
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Failed to correct text: {response.status_code} - {response.text}")


def text_to_speech(text):
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    text_chunks = split_text(text)

    audios = []  
    for i, chunk in enumerate(text_chunks):
        input_synthesis = texttospeech.SynthesisInput(text=chunk)
        
        voice = texttospeech.VoiceSelectionParams(language_code="en-US",name="en-US-Journey-D",)
        
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        
        response = client.synthesize_speech(input=input_synthesis, voice=voice, audio_config=audio_config)

        audio_path = f"output_chunk_{i}.wav"  
        with open(audio_path, "wb") as out:
            out.write(response.audio_content)

        audios.append(audio_path) 
    return audios 


#For syncing the video with new audio
def sync_audio(audio_path, target_duration):
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration

    if audio_duration < target_duration:
        loops_amount = int(np.ceil(target_duration / audio_duration))
        audio = concatenate_audioclips([audio] * loops_amount)
        audio = audio.subclip(0, target_duration) 
    elif audio_duration > target_duration:
        audio = audio.subclip(0, target_duration)

    altered_audio = "adjusted_output.wav"
    audio.write_audiofile(altered_audio)
    audio.close()

    return altered_audio

def replace_audio(video_path, audio_path, final_video):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    
    video = video.set_audio(audio)
    video.write_videofile(final_video, codec="libx264", audio_codec="aac")
    

def main():
    
    azure_openai_key = st.secrets["azure_openai_key"]
    azure_openai_endpoint = st.secrets["azure_openai_endpoint"]
    
    st.markdown("""<h1 style="text-align:center; color:#cc99ff; font-size: 48px;">🎬 Video Voice Converter</h1>""", unsafe_allow_html=True)
    st.markdown("""<h4 style="text-align:center; color:#ccccff;">Swap the audio with Journey voice</h4>""", unsafe_allow_html=True)
    st.markdown("""<h4 style="color:#ffffe6; font-size: 20px;"><i>[I am Anchita Sharma. Harkworking and love python. Super Excited and looking forward towards this opportunity.]</i></h4>""", unsafe_allow_html=True)
    st.markdown("""<h5 style="color:#ccccff;">Upload your video file:</h5>""", unsafe_allow_html=True)
    
    video_file = st.file_uploader("", type=["mp4"])

    if video_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        with st.spinner('Processing...'):
            audio = "extracted_audio.wav"
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio)
            
            transcripe = transcription(audio)
            corrected_text = text_correction(transcripe, azure_openai_key, azure_openai_endpoint)
            new_audio = text_to_speech(corrected_text)
            
            combined_audio = AudioSegment.empty()
            for audio in new_audio:
                chunk = AudioSegment.from_file(audio)
                combined_audio += chunk

            combined_audio_path = "combined_audio.wav"
            combined_audio.export(combined_audio_path, format="wav")

            original_duration = video_clip.audio.duration
            adjusted_audio = sync_audio(combined_audio_path, original_duration)

            output_video = "output_video.mp4"
            replace_audio(video_path, adjusted_audio, output_video)

        st.success("Your video has been successfully processed!")
        st.video(output_video)
        
    st.markdown("""<hr><p style="text-align:center;">Created by Anchita Sharma</p>""",unsafe_allow_html=True)

if __name__ == "__main__":
    main()
        