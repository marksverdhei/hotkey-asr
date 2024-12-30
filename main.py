from pynput.keyboard import Listener, Key
import winsound
import pyaudio
import wave
import threading
import pyperclip
from transformers import pipeline
import torch

# Configurable key combinations (using Key enum for special keys)
sound_keys = {Key.shift_l, Key.ctrl_l}  # Left Shift + F to start/stop recording
exit_keys = {'q', Key.ctrl_l}    # Left Ctrl + Q to exit

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

# MODEL = "openai/whisper-base"
MODEL = "NbAiLab/nb-whisper-base"


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

def normalize_key(key):
    """Convert key to a comparable format"""
    if isinstance(key, Key):
        return key
    if hasattr(key, 'char') and key.char is not None:
        return key.char.lower()
    return key

def check_key_combination(required_keys):
    """Check if all required keys are pressed"""
    normalized_pressed = {normalize_key(k) for k in pressed_keys}
    normalized_required = {normalize_key(k) for k in required_keys}
    return normalized_required.issubset(normalized_pressed)

def on_press(key):
    global RECORDING
    try:
        normalized_key = normalize_key(key)
        if normalized_key in {normalize_key(k) for k in sound_keys.union(exit_keys)}:
            pressed_keys.add(key)
            
            # Check for recording condition
            if not RECORDING and check_key_combination(sound_keys):
                play_enter_sound()
                start_recording()

            # Check for exit condition
            if check_key_combination(exit_keys):
                print(f"Exit key combination pressed. Exiting...")
                return False

    except Exception as e:
        print(f"Error handling key press: {e}")

def on_release(key):
    global RECORDING
    try:
        normalized_key = normalize_key(key)
        # Remove key from pressed_keys if it's there
        if key in pressed_keys:
            pressed_keys.remove(key)

            # Stop recording if any required sound key is released
            if RECORDING and normalized_key in {normalize_key(k) for k in sound_keys}:
                play_exit_sound()
                stop_recording()

    except Exception as e:
        print(f"Error handling key release: {e}")

# Format the key combinations for display
def format_key_combo(keys):
    return ' + '.join(str(k).replace('Key.', '') if isinstance(k, Key) else k.upper() for k in keys)

# Set up the listener
with Listener(on_press=on_press, on_release=on_release) as listener:
    print(f"Press {format_key_combo(sound_keys)} together to start recording.")
    print(f"Press {format_key_combo(exit_keys)} together to exit...")
    listener.join()