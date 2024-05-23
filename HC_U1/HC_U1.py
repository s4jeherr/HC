import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings

# Aufgabe 1

# Suppress the specific WavFileWarning
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# Read the WAV file
sample_rate, data = wavfile.read("Geheimnisvolle_Wellenlaengen.wav")

# Check if the audio file is stereo and take one channel if it is
if len(data.shape) > 1:
    data = data[:, 0]

# Select the block size
block_size = 16  # Change this value to select a different block size

# Initialize lists to store results
all_frequencies = []
all_magnitudes = []
main_freqs = []

# Loop through the data with a block size of `block_size` and a shift of 1 sample
num_blocks = len(data) - block_size + 1
for i in range(0, num_blocks):
    # Extract the current block of data
    block_data = data[i:i + block_size]

    # Perform the Fourier Transform on the current block
    audio_fft = np.fft.fft(block_data)
    frequencies = np.fft.fftfreq(block_size, d=1/sample_rate)

    # Take the magnitude of the FFT
    magnitude = np.abs(audio_fft)

    # Only consider the positive frequencies
    half_block_size = block_size // 2
    frequencies = frequencies[:half_block_size]
    magnitude = magnitude[:half_block_size]

    # Store the results
    all_frequencies.append(frequencies)
    all_magnitudes.append(magnitude)

    # Find the main frequencies and their amplitudes
    main_freq_idx = np.argmax(magnitude)
    main_freq = frequencies[main_freq_idx]
    main_amp = magnitude[main_freq_idx]
    main_freqs.append((main_freq, main_amp))

# Print the main frequencies and their amplitudes for each block
#print("Main Frequencies and their Amplitudes:")
#for i, (freq, amp) in enumerate(main_freqs):
    #print(f"Block {i}: Frequency = {freq:.2f} Hz, Amplitude = {amp:.2f}")

# Compute the frequency spectrum using FFT
audio_fft = np.fft.fft(data)
frequencies = np.fft.fftfreq(len(data), d=1/sample_rate)
magnitude = np.abs(audio_fft)

# Calculate statistics of the magnitude values
mean_magnitude = np.mean(magnitude)
std_magnitude = np.std(magnitude)

# Define a threshold for outlier detection
threshold = mean_magnitude + 3 * std_magnitude  # You can adjust the multiplier as needed

# Identify outlier frequencies
outlier_indices = np.where(magnitude > threshold)[0]
outlier_frequencies = frequencies[outlier_indices]
outlier_magnitudes = magnitude[outlier_indices]

# Plot the frequency spectrum
plt.figure(figsize=(12, 6))
plt.plot(frequencies, magnitude)
plt.plot(outlier_frequencies, outlier_magnitudes, 'ro')  # Plot outliers in red
plt.title('Frequency Spectrum with Outliers Highlighted')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
#plt.show()


# Aufgabe 2

# Size of input data (in bytes)
data_size_bytes = data.nbytes

# Size of FFT results (complex numbers, each complex number has real and imaginary parts)
fft_results_size_bytes = 2 * len(data) * np.dtype(np.complex64).itemsize

# Size of each frequency value (in bytes)
frequency_value_size_bytes = np.dtype(frequencies.dtype).itemsize

# Size of each magnitude value (in bytes)
magnitude_value_size_bytes = np.dtype(magnitude.dtype).itemsize

# Number of frequency values (same as number of magnitude values)
num_values = len(frequencies)

# Memory usage of the frequencies array
frequencies_array_size_bytes = frequency_value_size_bytes * num_values

# Memory usage of the magnitudes array
magnitudes_array_size_bytes = magnitude_value_size_bytes * num_values

# Total memory usage of intermediate arrays
intermediate_arrays_size_bytes = frequencies_array_size_bytes + magnitudes_array_size_bytes

# Total memory usage
total_memory_bytes = data_size_bytes + fft_results_size_bytes + intermediate_arrays_size_bytes

# Convert to megabytes (MB)
total_memory_mb = total_memory_bytes / (1024 * 1024)

print(f"Total memory usage: {total_memory_mb:.2f} MB")

