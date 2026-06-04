# test_mic.py
import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    rms = np.sqrt(np.mean(indata**2))
    barras = int(rms * 1000)
    print(f"RMS: {rms:.4f}  {'█' * min(barras, 50)}")

print("Habla al micrófono... (Ctrl+C para salir)")
with sd.InputStream(samplerate=16000, channels=1, callback=callback, blocksize=512):
    sd.sleep(10000)