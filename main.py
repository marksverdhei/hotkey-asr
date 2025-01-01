from pynput.keyboard import Listener as KeyboardListener, Key
from pynput.mouse import Listener as MouseListener, Button
import winsound
import pyaudio
import wave
import threading
import pyperclip
from transformers import pipeline
import torch
import yaml


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

MODEL = "openai/whisper-base"
DEVICE = -1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transcriber = pipeline(
    task="automatic-speech-recognition",
    model=MODEL,
    device=DEVICE
)

print(transcriber.model.device)

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

def normalize_trigger(trigger):
    """Convert keyboard key or mouse button to a comparable format"""
    if isinstance(trigger, (Key, Button)):
        return trigger
    if hasattr(trigger, 'char') and trigger.char is not None:
        return trigger.char.lower()
    return trigger

def check_trigger_combination(required_triggers):
    """Check if all required triggers are pressed"""
    normalized_pressed = {normalize_trigger(t) for t in pressed_triggers}
    normalized_required = {normalize_trigger(t) for t in required_triggers}
    return normalized_required.issubset(normalized_pressed)

def on_press(key):
    try:
        normalized_key = normalize_trigger(key)
        if normalized_key in {normalize_trigger(k) for k in sound_triggers.union(exit_triggers)}:
            pressed_triggers.add(key)
            
            # Check for recording condition
            if not RECORDING and check_trigger_combination(sound_triggers):
                play_enter_sound()
                start_recording()

            # Check for exit condition
            if check_trigger_combination(exit_triggers):
                print(f"Exit combination pressed. Exiting...")
                return False

    except Exception as e:
        print(f"Error handling key press: {e}")

def on_release(key):
    try:
        normalized_key = normalize_trigger(key)
        # Remove key from pressed_triggers if it's there
        if key in pressed_triggers:
            pressed_triggers.remove(key)

            # Stop recording if any required sound trigger is released
            if RECORDING and normalized_key in {normalize_trigger(t) for t in sound_triggers}:
                play_exit_sound()
                stop_recording()

    except Exception as e:
        print(f"Error handling key release: {e}")

def on_click(x, y, button, pressed):
    try:
        if pressed:
            if button in {normalize_trigger(t) for t in sound_triggers}:
                pressed_triggers.add(button)
                
                # Check for recording condition
                if not RECORDING and check_trigger_combination(sound_triggers):
                    play_enter_sound()
                    start_recording()
        else:  # Released
            if button in pressed_triggers:
                pressed_triggers.remove(button)
                
                # Stop recording if any required sound trigger is released
                if RECORDING and button in {normalize_trigger(t) for t in sound_triggers}:
                    play_exit_sound()
                    stop_recording()
    except Exception as e:
        print(f"Error handling mouse click: {e}")

# Format the trigger combinations for display
def format_trigger_combo(triggers):
    return ' + '.join(str(t).replace('Key.', '').replace('Button.', '') 
                     if isinstance(t, (Key, Button)) else t.upper() 
                     for t in triggers)

# Set up both listeners
keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release)
mouse_listener = MouseListener(on_click=on_click)

# Start both listeners
keyboard_listener.start()
mouse_listener.start()

print(f"Press {format_trigger_combo(sound_triggers)} together to start recording.")
print(f"Press {format_trigger_combo(exit_triggers)} together to exit...")

# Keep the program running until the keyboard listener stops
keyboard_listener.join()