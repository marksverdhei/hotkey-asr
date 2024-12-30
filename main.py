from pynput.keyboard import Listener
from pynput.keyboard import Key
import winsound
import pyaudio
import wave
import threading
import pyperclip
from transformers import pipeline
import torch

# Configurable key combinations
# sound_keys = {Key.shift, 'f'}
sound_keys = {'f'}

exit_keys = {'j', 'l'}

pressed_keys = set()

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORDING = False
frames = []
stream = None
p = None

# Initialize the Whisper model pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base",
    device=0 if device == 'cuda' else -1
)

# Functions for microphone recording
def start_recording():
    global RECORDING, frames, stream, p
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    RECORDING = True
    frames = []

    def record():
        while RECORDING:
            data = stream.read(CHUNK)
            frames.append(data)

    threading.Thread(target=record, daemon=True).start()
    print("Recording started...")

def stop_recording():
    global RECORDING, stream, p
    RECORDING = False
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save recording to a file
    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording stopped and saved to output.wav")

    # Transcribe the audio and copy to clipboard
    transcribe_and_copy("output.wav")

def transcribe_and_copy(audio_file):
    print("Transcribing audio...")
    result = transcriber(audio_file)
    transcription = result['text']
    print("Transcription:", transcription)

    # Copy transcription to clipboard
    pyperclip.copy(transcription)
    print("Transcription copied to clipboard.")
    play_loaded_sound()

def play_enter_sound():
    frequency = 440  # Frequency in Hertz
    duration = 100  # Duration in milliseconds
    winsound.Beep(frequency, duration)

def play_exit_sound():
    frequency = 220  # Frequency in Hertz
    duration = 100  # Duration in milliseconds
    winsound.Beep(frequency, duration)

def play_loaded_sound():
    frequency = 660  # Frequency in Hertz
    duration = 100  # Duration in milliseconds
    winsound.Beep(frequency, duration)


def on_press(key):
    global RECORDING
    try:
        # Handle character keys
        if key.char in sound_keys or key.char in exit_keys:
            if key.char not in pressed_keys:
                pressed_keys.add(key.char)
                print(f"Key pressed: {key.char}")

                # Play sound and start recording for sound keys
                if key.char in sound_keys:
                    play_enter_sound()
                    if not RECORDING:
                        start_recording()

                # Check if all keys in exit_keys are pressed
                if exit_keys.issubset(pressed_keys):
                    print(f"{exit_keys} pressed together. Exiting...")
                    return False
    except AttributeError:
        # Handle special keys (ignored in this case)
        pass

def on_release(key):
    global RECORDING
    try:
        # Handle character keys
        if key.char in sound_keys or key.char in exit_keys:
            if key.char in pressed_keys:
                pressed_keys.remove(key.char)
                print(f"Key released: {key.char}")
                play_exit_sound()

                # Stop recording for sound keys
                if key.char in sound_keys and RECORDING:
                    stop_recording()
    except AttributeError:
        # Handle special keys (ignored in this case)
        pass

# Set up the listener
with Listener(on_press=on_press, on_release=on_release) as listener:
    print("Press 'f' to start/stop recording. Press 'j' and 'l' together to exit...")
    listener.join()
