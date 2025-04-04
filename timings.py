import os
import time
from transformers import pipeline
from faster_whisper import WhisperModel
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to -1 to use CPU


hf_whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base", return_timestamps=True, device="cuda")
faster_model = WhisperModel("base", device="cuda")  # You can use other sizes like "medium" or "large-v2"


def time_huggingface_whisper(audio_file):
    print("Timing Hugging Face Whisper...")
    start_time = time.time()

    # Load Hugging Face Whisper pipeline

    # Transcribe audio file
    transcription = hf_whisper(audio_file)
    end_time = time.time()

    print("Hugging Face Whisper transcription:", transcription["text"])
    print("Hugging Face Whisper time taken:", end_time - start_time, "seconds")

    return end_time - start_time

def time_faster_whisper(audio_file):
    print("Timing Faster Whisper...")
    start_time = time.time()

    # Load Faster Whisper model

    # Transcribe audio file
    segments, _ = faster_model.transcribe(audio_file, )
    transcription = "".join(segment.text for segment in segments)
    end_time = time.time()

    print("Faster Whisper transcription:", transcription)
    print("Faster Whisper time taken:", end_time - start_time, "seconds")

    return end_time - start_time

def main():
    audio_file = "noice.mp3"  # Replace with the path to your audio file

    # Measure time for Hugging Face Whisper
    hf_time = time_huggingface_whisper(audio_file)

    # Measure time for Faster Whisper
    fw_time = time_faster_whisper(audio_file)

    # Compare results
    print("\nComparison:")
    print(f"Hugging Face Whisper time: {hf_time:.2f} seconds")
    print(f"Faster Whisper time: {fw_time:.2f} seconds")

    if hf_time < fw_time:
        print("Hugging Face Whisper is faster.")
    else:
        print("Faster Whisper is faster.")

if __name__ == "__main__":
    main()
