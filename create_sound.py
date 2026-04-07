import numpy as np
from scipy.io import wavfile

# Generate a high-pitched beep sound (e.g., 1000 Hz)
sample_rate = 44100
duration = 0.5  # seconds
frequency = 800.0  # Hz

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Add an envelope to avoid clicks at the beginning/end
envelope = np.ones_like(t)
decay_time = 0.05
decay_samples = int(sample_rate * decay_time)
envelope[:decay_samples] = np.linspace(0, 1, decay_samples)
envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
audio *= envelope

# Normalize to 16-bit range and write to file
audio = np.int16(audio * 32767)
wavfile.write("assets/alert.wav", sample_rate, audio)
print("Alert sound generated at assets/alert.wav")
