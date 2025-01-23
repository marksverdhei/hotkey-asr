import os
import winsound
import pyaudio
import wave
import threading
import torch
import yaml
import io
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import dotenv
dotenv.load_dotenv(".env")
# If you have a standard openai package:
#   pip install openai
# then do:
# import openai
#
# If you have a custom "OpenAI" class (from your snippet #3):
#   from openai import OpenAI
# and later instantiate it with:
#   client = OpenAI()
#
# For simplicity here, I'll use the snippet #3 style:
from openai import OpenAI

from pynput.keyboard import Listener as KeyboardListener, Key
from pynput.mouse import Listener as MouseListener, Button
from transformers import pipeline

def convert(l) -> set:
    return {eval(i) if "." in i else i for i in l} 

config_path = "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f.read())

hotkeys = config["hotkeys"]
sound_triggers = convert(hotkeys["record"])
exit_triggers = convert(hotkeys["exit"])
pressed_triggers = set()

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORDING = False
frames = []
stream = None
p = None

MODEL = config["model"]
DEVICE = config["device"]

# Initialize the ASR pipeline
transcriber = pipeline(
    task="automatic-speech-recognition",
    model=MODEL,
    device=DEVICE
)

# If using the standard openai library, you'd do something like:
# openai.api_key = config["openai_api_key"]

# Using your snippet #3 style:
print(os.getenv("OPENAI_API_KEY")[-5:])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

############################################################
#                   UTILITY FUNCTIONS                      #
############################################################

def play_enter_sound():
    """Beep when recording starts."""
    frequency = 440
    duration = 100
    winsound.Beep(frequency, duration)

def play_exit_sound():
    """Beep when recording stops."""
    frequency = 220
    duration = 100
    winsound.Beep(frequency, duration)

def play_tts_beep():
    """Beep to indicate TTS audio is being played to VB cable."""
    frequency = 660
    duration = 100
    winsound.Beep(frequency, duration)

def start_recording():
    """Start recording from default mic in a background thread."""
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
    """Stop recording and transcribe the audio."""
    global RECORDING, stream, p
    if not RECORDING:
        return
        
    RECORDING = False
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Create in-memory WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    # Convert WAV to numpy array for the model
    wav_buffer.seek(0)
    with wave.open(wav_buffer, 'rb') as wf:
        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        
    transcribe_and_tts(audio_data)

def transcribe_and_tts(audio_data):
    """Transcribe audio using the ASR model, then send it to OpenAI TTS, then play the TTS output."""
    print("Transcribing audio...")
    result = transcriber({"sampling_rate": RATE, "raw": audio_data})
    transcription = result['text'].strip()
    print("Transcription:", transcription)

    # Now perform TTS with OpenAI
    speech_file_path = tts_openai(transcription)

    # Finally, play the TTS audio into VB Cable
    play_tts_audio(speech_file_path)

def tts_openai(text):
    """
    Use OpenAI's TTS to synthesize speech.
    This uses your snippet #3 pattern:
        response = client.audio.speech.create(...)
        response.stream_to_file(...)
    Returns the path to the generated .mp3 file.
    """
    # You may need to adapt the voice/model per your TTS subscription or usage:
    speech_file_path = "tts_output.mp3"
    response = client.audio.speech.create(
        model="tts-1",   # adjust if needed
        voice="nova",   # adjust if needed
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

def play_tts_audio(mp3_path):
    """Play the TTS mp3 file through the VB Cable input device."""
    # First beep to indicate TTS playback
    play_tts_beep()

    # Load audio file
    audio = AudioSegment.from_file(mp3_path, format="mp3")

    # Convert to NumPy array
    raw_data = np.frombuffer(audio.raw_data, dtype=np.int16)

    # Get audio properties
    sample_rate = audio.frame_rate
    channels = audio.channels

    # Reshape the raw data for multi-channel audio if needed
    if channels > 1:
        raw_data = raw_data.reshape((-1, channels))

    # Find VB Cable device
    device_name = "CABLE Input"
    devices = sd.query_devices()
    vb_device = next((d for d in devices if device_name in d['name']), None)

    if vb_device:
        device_index = vb_device['index']
        print(f"Playing TTS audio on device: {vb_device['name']}")
        sd.play(raw_data, samplerate=sample_rate, device=device_index)
        sd.wait()
    else:
        print("VB Cable device not found. Check the device name and try again.")

############################################################
#                  HOTKEY/MOUSE LISTENERS                  #
############################################################

def normalize_trigger(trigger):
    """
    Convert a key or button press object to a common form
    so it can be compared to config triggers accurately.
    """
    if isinstance(trigger, (Key, Button)):
        return trigger
    if hasattr(trigger, 'char') and trigger.char is not None:
        return trigger.char.lower()
    return trigger

def check_trigger_combination(required_triggers):
    """
    Check if all keys/buttons in required_triggers
    are currently pressed (in pressed_triggers).
    """
    normalized_pressed = {normalize_trigger(t) for t in pressed_triggers}
    normalized_required = {normalize_trigger(t) for t in required_triggers}
    return normalized_required.issubset(normalized_pressed)

def on_press(key):
    try:
        normalized_key = normalize_trigger(key)
        # Check if this key is part of the triggers we care about
        if normalized_key in {normalize_trigger(k) for k in sound_triggers.union(exit_triggers)}:
            pressed_triggers.add(key)
            
            # If the record trigger combination is fully pressed, start recording
            if not RECORDING and check_trigger_combination(sound_triggers):
                play_enter_sound()
                start_recording()

            # If the exit trigger combination is pressed, stop everything
            if check_trigger_combination(exit_triggers):
                print("Exit combination pressed. Exiting...")
                return False

    except Exception as e:
        print(f"Error handling key press: {e}")

def on_release(key):
    try:
        normalized_key = normalize_trigger(key)
        if key in pressed_triggers:
            pressed_triggers.remove(key)

            # If we were recording and any part of the record combination is released, stop recording
            if RECORDING and normalized_key in {normalize_trigger(t) for t in sound_triggers}:
                play_exit_sound()
                stop_recording()

    except Exception as e:
        print(f"Error handling key release: {e}")

def on_click(x, y, button, pressed):
    """Optional: handle mouse clicks as triggers, if configured."""
    try:
        if pressed:
            if button in {normalize_trigger(t) for t in sound_triggers}:
                pressed_triggers.add(button)
                if not RECORDING and check_trigger_combination(sound_triggers):
                    play_enter_sound()
                    start_recording()
        else:
            if button in pressed_triggers:
                pressed_triggers.remove(button)
                if RECORDING and button in {normalize_trigger(t) for t in sound_triggers}:
                    play_exit_sound()
                    stop_recording()
    except Exception as e:
        print(f"Error handling mouse click: {e}")

def format_trigger_combo(triggers):
    """Helper function to display triggers nicely in console."""
    return ' + '.join(str(t).replace('Key.', '').replace('Button.', '') 
                     if isinstance(t, (Key, Button)) else t.upper() 
                     for t in triggers)

############################################################
#                       MAIN THREAD                        #
############################################################

keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release)
mouse_listener = MouseListener(on_click=on_click)

keyboard_listener.start()
mouse_listener.start()

print(f"Press {format_trigger_combo(sound_triggers)} together to start recording.")
print(f"Press {format_trigger_combo(exit_triggers)} together to exit...")

keyboard_listener.join()
