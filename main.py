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
sound_keys = {"f", "g"}  # Must press both simultaneously
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

MODEL = "openai/whisper-base"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transcriber = pipeline(
    task="automatic-speech-recognition",
    model=MODEL,
    device=0 if device == 'cuda' else -1
)

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
    if not RECORDING:  # Don't stop if not recording
        return
        
    RECORDING = False
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording stopped and saved to output.wav")

    transcribe_and_copy("output.wav")

def transcribe_and_copy(audio_file):
    print("Transcribing audio...")
    result = transcriber(audio_file)
    transcription = result['text']
    print("Transcription:", transcription)
    pyperclip.copy(transcription)
    print("Transcription copied to clipboard.")
    play_loaded_sound()

def play_enter_sound():
    frequency = 440
    duration = 100
    winsound.Beep(frequency, duration)

def play_exit_sound():
    frequency = 220
    duration = 100
    winsound.Beep(frequency, duration)

def play_loaded_sound():
    frequency = 660
    duration = 100
    winsound.Beep(frequency, duration)

def check_keys():
    """Debug function to print current key state"""
    print(f"Current pressed keys: {pressed_keys}")
    print(f"Required sound keys: {sound_keys}")
    print(f"All required keys pressed? {sound_keys.issubset(pressed_keys)}")
    print(f"Extra keys pressed? {pressed_keys - sound_keys}")

def on_press(key):
    global RECORDING
    try:
        # Convert key to lowercase string if it's a character
        if hasattr(key, 'char') and key.char is not None:
            key_char = key.char.lower()
        else:
            return  # Ignore non-character keys
            
        # Add key to pressed_keys if it's a sound key or exit key
        if key_char in sound_keys or key_char in exit_keys:
            pressed_keys.add(key_char)
            # print(f"\nKey pressed: {key_char}")
            # check_keys()  # Debug print

            # Check for recording condition: all sound keys must be pressed
            if not RECORDING and sound_keys.issubset(pressed_keys):
                # Only start if we have exactly the sound keys pressed
                if pressed_keys.intersection(sound_keys) == sound_keys:
                    play_enter_sound()
                    start_recording()

            # Check for exit condition
            if exit_keys.issubset(pressed_keys):
                print(f"{exit_keys} pressed together. Exiting...")
                return False

    except AttributeError as e:
        print(f"Error handling key press: {e}")
        pass

def on_release(key):
    global RECORDING
    try:
        # Convert key to lowercase string if it's a character
        if hasattr(key, 'char') and key.char is not None:
            key_char = key.char.lower()
        else:
            return  # Ignore non-character keys

        # Remove key from pressed_keys if it's there
        if key_char in pressed_keys:
            pressed_keys.remove(key_char)
            print(f"\nKey released: {key_char}")
            # check_keys()  # Debug print

            # Stop recording if any required sound key is released
            if RECORDING and key_char in sound_keys:
                play_exit_sound()
                stop_recording()

    except AttributeError as e:
        print(f"Error handling key release: {e}")
        pass

# Set up the listener
with Listener(on_press=on_press, on_release=on_release) as listener:
    print(f"Press {'+'.join(sound_keys)} together to start recording.")
    print(f"Press {'+'.join(exit_keys)} together to exit...")
    listener.join()