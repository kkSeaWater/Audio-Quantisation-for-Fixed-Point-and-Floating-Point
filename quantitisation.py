import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import firwin, lfilter

#Load audio file
audio_path = r"C:\Users\User\Desktop\SSE\DCA\Ass-2\codes\dog1.wav"  
audio_data, sample_rate = sf.read(audio_path)
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]  # Convert stereo to mono

# Design FIR low-pass filter
def design_fir_filter(cutoff_freq, fs, num_taps=31):
    nyquist = fs / 2
    norm_cutoff = cutoff_freq / nyquist
    return firwin(num_taps, norm_cutoff)

b = design_fir_filter(1000, sample_rate)

# Fixed-point simulation function
def float_to_fixed(x, int_bits, frac_bits):
    scale = 2 ** frac_bits
    max_val = (2 ** (int_bits + frac_bits - 1)) - 1
    min_val = -1 * (2 ** (int_bits + frac_bits - 1))
    x_fixed = np.round(x * scale).astype(np.int32)
    x_clipped = np.clip(x_fixed, min_val, max_val)
    return x_clipped / scale

# Apply FIR filter in 4 formats
filtered_outputs = {}

# 1. Float32
signal_f32 = audio_data.astype(np.float32)
b_f32 = b.astype(np.float32)
filtered_outputs["float32"] = lfilter(b_f32, 1.0, signal_f32)

# 2. Float16
signal_f16 = audio_data.astype(np.float16)
b_f16 = b.astype(np.float16)
filtered_outputs["float16"] = lfilter(b_f16, 1.0, signal_f16)

# 3. Fixed-point Q1.7
sig_q1_7 = float_to_fixed(audio_data, 1, 7)
b_q1_7 = float_to_fixed(b, 1, 7)
filtered_outputs["Q1.7"] = lfilter(b_q1_7, 1.0, sig_q1_7)

# 4. Fixed-point Q3.5
sig_q3_5 = float_to_fixed(audio_data, 3, 5)
b_q3_5 = float_to_fixed(b, 3, 5)
filtered_outputs["Q3.5"] = lfilter(b_q3_5, 1.0, sig_q3_5)

# Compute error metrics
def compute_mse(ref, test): return np.mean((ref - test) ** 2)
def compute_snr(ref, test):
    noise = ref - test
    return 10 * np.log10(np.mean(ref ** 2) / np.mean(noise ** 2)) if np.mean(noise ** 2) > 0 else np.inf

mse = {}
snr = {}
errors = {}
reference = filtered_outputs["float32"]

for k in ["float16", "Q1.7", "Q3.5"]:
    errors[k] = reference - filtered_outputs[k]
    mse[k] = compute_mse(reference, filtered_outputs[k])
    snr[k] = compute_snr(reference, filtered_outputs[k])

# Plot filter outputs
plt.figure(figsize=(12, 6))
for label, signal in filtered_outputs.items():
    plt.plot(signal[:1000], label=label, linewidth=2 if label == "float32" else 1, linestyle='-' if label == "float32" else '--')
plt.title("FIR filter outputs (float32, float16, Q1.7, Q3.5)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot error signals
plt.figure(figsize=(12, 4))
for k, err in errors.items():
    plt.plot(err[:1000], label=f"Error: {k}")
plt.title("Error compared to float32")
plt.xlabel("Samples")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print MSE and SNR
print("\n==== Mean Squared Error (MSE) ====")
for k, v in mse.items():
    print(f"{k}: {v:.6e}")

print("\n==== Signal-to-Noise Ratio (SNR) ====")
for k, v in snr.items():
    print(f"{k}: {v:.2f} dB")
