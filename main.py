from contextlib import contextmanager
import time
import winsound
from faster_whisper import WhisperModel
import pyaudio
import wave
import threading
import torch
import yaml
import io
import numpy as np
import sounddevice as sd
def get_virtual_cable_device_index():
    device_name = "CABLE Input"
    devices = sd.query_devices()
    vb_device = next((d for d in devices if device_name in d['name']), None)

    if vb_device:
        return vb_device['index']
    else:
        print("VB Cable device not found. Check the device name and try again.")
        return None
    
vc = get_virtual_cable_device_index()
sd.default.device = (vc, vc)
from pydub import AudioSegment

from pynput.keyboard import Listener as KeyboardListener, Key
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Listener as MouseListener, Button
from transformers import pipeline
import os
import dotenv

from openai import OpenAI
import elevenlabs

############################################################
#                 CONFIG & GLOBAL VARIABLES               #
############################################################
dotenv.load_dotenv(".env")




def convert(l) -> set:
    return {eval(i) if "." in i else i for i in l}

config_path = "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f.read())

hotkeys = config["hotkeys"]
sound_triggers = convert(hotkeys["record"])
exit_triggers = convert(hotkeys["exit"])
pressed_triggers = set()

DEVICE = config["device"]
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE  # Set to -1 to use CPU
VOICE_PROFILE = config["voice_profile"]

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44100
RATE = 16000

RECORDING = False
frames = []
stream = None
p = None
PLAY_PADDING = 0.5  # seconds

MODEL = config["model"]
LOCAL = config["local"]
elabs_voice_id = config["elabs_voice_id"]
tts_oai = False

elevenlabs_client = elevenlabs.ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def tts_openai(text):
    speech_file_path = "tts_output.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice=VOICE_PROFILE,
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

def tts_elevenlabs(text):
    audio = elevenlabs_client.text_to_speech.convert(
        text=text,
        optimize_streaming_latency=3,
        voice_id=elabs_voice_id,
        model_id="eleven_flash_v2",
        output_format="mp3_44100_128",
    )

    return audio
if tts_oai:
    tts_func = tts_openai
else:
    tts_func = tts_elevenlabs

# Initialize the ASR pipeline
# transcriber = pipeline(
#     task="automatic-speech-recognition",
#     model=MODEL,
#     device=DEVICE
# )

faster_model = WhisperModel("base", device="cuda" if int(DEVICE) >= 0 else "cpu")
print(os.getenv("OPENAI_API_KEY")[-5:])

client = OpenAI()

# Optional: parse the key string from the config
key_during_tts_str = config.get("key_during_tts", None)  # e.g. "Key.ctrl_l" or "a"

def parse_key_str(key_str):
    """
    Convert a string like 'Key.ctrl_l' or 'a'
    into a pynput.keyboard.Key or a character.
    """
    if not key_str:
        return None
    
    if key_str.startswith("Key."):
        # e.g. "Key.ctrl_l" -> Key.ctrl_l
        subkey = key_str[4:]
        return getattr(Key, subkey, None)
    else:
        # Single character
        return key_str

key_during_tts = parse_key_str(key_during_tts_str)

keyboard_controller = KeyboardController()

############################################################
#                   UTILITY FUNCTIONS                      #
############################################################

def play_enter_sound():
    frequency = 440
    duration = 100
    winsound.Beep(frequency, duration)

def play_exit_sound():
    frequency = 220
    duration = 100
    winsound.Beep(frequency, duration)

def play_asr_beep():
    frequency = 880
    duration = 100
    winsound.Beep(frequency, duration)

def play_tts_beep():
    frequency = 660
    duration = 100
    winsound.Beep(frequency, duration)

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
    transcribe(wav_buffer)
    

def transcribe(wav_buffer):
    if LOCAL:
        with wave.open(wav_buffer, 'rb') as wf:

            _transcribe_local(wf)
    else:
        _transcribe_oai(wav_buffer)
    
def _transcribe_oai(wf: wave.Wave_read):
    setattr(wf, 'name', 'audio.wav')
    threading.Thread(
        target=_transcribe_tts_oai_daemon, 
        args=(wf,), 
        daemon=True
    ).start()

def _transcribe_tts_oai_daemon(wf: wave.Wave_read):
    transcription_data = client.audio.transcriptions.create(
        model="whisper-1", 
        file=wf,
    )
    transcription = transcription_data.text.strip()

    # Now perform TTS with OpenAI
    speech_file_path = tts_func(transcription)
    play_asr_beep()

    # Finally, play the TTS audio in background (also non-blocking).
    threading.Thread(
        target=play_tts_audio, 
        args=(speech_file_path,),
        daemon=True
    ).start()

def _transcribe_local(wf: wave.Wave_read):
    audio_data = np.frombuffer(
        wf.readframes(wf.getnframes()), 
        dtype=np.int16
    ).astype(np.float32) / 32768.0
    
    # Transcribe in the same callback
    # Then do TTS *in a separate thread*:
    # transcribe_and_tts(audio_data)
    threading.Thread(
        target=transcribe_and_tts, 
        args=(audio_data,), 
        daemon=True
    ).start()


@contextmanager
def hold_key(key):
    print(f"Holding down key: {key}")
    keyboard_controller.press(key)
    yield
    time.sleep(PLAY_PADDING)
    print(f"Releasing key: {key}")
    keyboard_controller.release(key)

def transcribe_and_tts(audio_data):
    print("Transcribing audio...")
    # result = transcriber({"sampling_rate": RATE, "raw": audio_data})
    result = faster_model.transcribe(audio_data)
    segments, info = result
    transcription = "".join(segment.text for segment in segments).strip()

    print("Transcription:", transcription)
    if not transcription or transcription.lower() == "you":
        print("No transcription found.")
        return
    
    play_asr_beep()
    # Now perform TTS with OpenAI
    audio = tts_elevenlabs(transcription)

    with hold_key(key_during_tts):
        elevenlabs.play(audio, use_ffmpeg=False)



def play_tts_audio(mp3_path):
    play_tts_beep()

    # Optionally hold down a key. Remember: this can cause
    # the "locked" feeling if it's a modifier like Ctrl/Alt.
    if key_during_tts is not None:
        print(f"Holding down key: {key_during_tts_str}")
        keyboard_controller.press(key_during_tts)

    try:
        audio = AudioSegment.from_file(mp3_path, format="mp3")
        raw_data = np.frombuffer(audio.raw_data, dtype=np.int16)
        sample_rate = audio.frame_rate
        channels = audio.channels

        if channels > 1:
            raw_data = raw_data.reshape((-1, channels))

        device_name = "CABLE Input"
        devices = sd.query_devices()
        vb_device = next((d for d in devices if device_name in d['name']), None)

        if vb_device:
            device_index = vb_device['index']
            print(f"Playing TTS audio on device: {vb_device['name']}")
            sd.play(raw_data, samplerate=sample_rate, device=device_index)
            sd.wait()  # Will block *this* thread, not the main thread
            time.sleep(PLAY_PADDING)
        else:
            print("VB Cable device not found. Check the device name and try again.")
    finally:
        if key_during_tts is not None:
            print(f"Releasing key: {key_during_tts_str}")
            keyboard_controller.release(key_during_tts)

############################################################
#                  HOTKEY/MOUSE LISTENERS                  #
############################################################

def normalize_trigger(trigger):
    if isinstance(trigger, (Key, Button)):
        return trigger
    if hasattr(trigger, 'char') and trigger.char is not None:
        return trigger.char.lower()
    return trigger

def check_trigger_combination(required_triggers):
    normalized_pressed = {normalize_trigger(t) for t in pressed_triggers}
    normalized_required = {normalize_trigger(t) for t in required_triggers}
    return normalized_required.issubset(normalized_pressed)

def on_press(key):
    try:
        normalized_key = normalize_trigger(key)
        if normalized_key in {normalize_trigger(k) for k in sound_triggers.union(exit_triggers)}:
            pressed_triggers.add(key)
            
            if not RECORDING and check_trigger_combination(sound_triggers):
                play_enter_sound()
                start_recording()

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
    return ' + '.join(
        str(t).replace('Key.', '').replace('Button.', '') 
        if isinstance(t, (Key, Button)) else t.upper() 
        for t in triggers
    )

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
