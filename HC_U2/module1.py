import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
from concurrent.futures import ProcessPoolExecutor
from numba import cuda, jit, float32
from math import ceil
import cupy as cp
from pyculib.fft import FFT


def sequential(wav_path, block_size, offset, threshold):
    # Validate the block size
    if not (64 <= block_size <= 512):
        raise ValueError("Block size must be between 64 and 512")

    # Validate the offset
    if offset >= block_size:
        raise ValueError("Offset must be smaller than the block size")

    # Read the WAV file
    sample_rate, data = wavfile.read(wav_path)

    # If the data is stereo, convert it to mono by averaging the channels
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Initialize a list to hold the FFT results
    fft_results = []

    # Iterate over the data in blocks
    for start in range(0, len(data) - block_size + 1, offset):
        block = data[start:start + block_size]
        fft_result = np.fft.rfft(block)
        fft_magnitude = np.abs(fft_result)
        fft_results.append(fft_magnitude)

    # Convert the list of FFT results to a numpy array for easier manipulation
    fft_results = np.array(fft_results)

    # Compute the mean amplitude for each frequency
    mean_amplitudes = np.mean(fft_results, axis=0)

    # Find and print the frequencies with mean amplitude above the threshold
    frequencies = np.fft.rfftfreq(block_size, d=1/sample_rate)
    for freq, amp in zip(frequencies, mean_amplitudes):
        if amp > threshold:
            print(f"Frequency: {freq:.2f} Hz, Mean Amplitude: {amp:.2f}")


def generate_wav_file(filename, frequency, amplitude, duration, sample_rate=44100):
    """
    Generate a WAV file with a sine wave signal.

    Parameters:
        filename (str): The name of the output WAV file.
        frequency (float): The frequency of the sine wave in Hz.
        amplitude (float): The amplitude of the sine wave (0.0 to 1.0).
        duration (float): The duration of the signal in seconds.
        sample_rate (int): The sample rate in Hz. Default is 44100.
    """
    # Generate the time axis for the signal
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate the sine wave signal
    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Ensure the signal is in the range [-1, 1]
    signal = np.clip(signal, -1, 1)

    # Convert the signal to 16-bit PCM format
    pcm_signal = np.int16(signal * 32767)

    # Write the signal to a WAV file
    write(filename, sample_rate, pcm_signal)

def process_blocks(data, starts, block_size):
    fft_results = []
    for start in starts:
        block = data[start:start + block_size]
        fft_result = np.fft.rfft(block)
        fft_magnitude = np.abs(fft_result)
        fft_results.append(fft_magnitude)
    return fft_results

def parallel_cpu(wav_path, block_size, offset, threshold, num_cores):
    # Validate the block size
    if not (64 <= block_size <= 512):
        raise ValueError("Block size must be between 64 and 512")

    # Validate the offset
    if offset >= block_size:
        raise ValueError("Offset must be smaller than the block size")

    # Read the WAV file
    sample_rate, data = wavfile.read(wav_path)

    # If the data is stereo, convert it to mono by averaging the channels
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Define the start points for each block
    starts = list(range(0, len(data) - block_size + 1, offset))

    # Split start points into chunks for each core
    chunk_size = (len(starts) + num_cores - 1) // num_cores
    chunks = [starts[i:i + chunk_size] for i in range(0, len(starts), chunk_size)]

    # Initialize a list to hold the FFT results
    fft_results = []

    # Use ProcessPoolExecutor to parallelize the processing of blocks
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_blocks, data, chunk, block_size) for chunk in chunks]

        for future in futures:
            fft_results.extend(future.result())

    # Convert the list of FFT results to a numpy array for easier manipulation
    fft_results = np.array(fft_results)

    # Compute the mean amplitude for each frequency
    mean_amplitudes = np.mean(fft_results, axis=0)

    # Find and print the frequencies with mean amplitude above the threshold
    frequencies = np.fft.rfftfreq(block_size, d=1/sample_rate)
    for freq, amp in zip(frequencies, mean_amplitudes):
        if amp > threshold:
            print(f"Frequency: {freq:.2f} Hz, Mean Amplitude: {amp:.2f}")

def compare_times_cpu(wav_path, block_size, offset, threshold, num_cores):
    # Measure time for the original function
    start_time = time.time()
    sequential(wav_path, block_size, offset, threshold)
    original_time = time.time() - start_time

    # Measure time for the parallel function
    start_time = time.time()
    parallel_cpu(wav_path, block_size, offset, threshold, num_cores)
    parallel_time = time.time() - start_time

    # Calculate speedup
    speedup = original_time / parallel_time

    print(f"Original function time: {original_time:.4f} seconds")
    print(f"Parallel function time: {parallel_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")

@cuda.jit
def process_blocks(data, starts, block_size, results):
    thread_id = cuda.grid(1)
    if thread_id < len(starts):
        start = starts[thread_id]
        for i in range(block_size):
            results[thread_id, i] = data[start + i]

def parallel_gpu(wav_path, block_size, offset, threshold, num_cores):
    # Validate the block size and offset
    if not (64 <= block_size <= 512):
        raise ValueError("Block size must be between 64 and 512")
    if offset >= block_size:
        raise ValueError("Offset must be smaller than the block size")

    # Read the WAV file
    sample_rate, data = wavfile.read(wav_path)

    # If the data is stereo, convert it to mono by averaging the channels
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Define the start points for each block
    starts = np.arange(0, len(data) - block_size + 1, offset)

    # Allocate GPU memory for results
    fft_results_real = np.zeros((len(starts), block_size), dtype=np.float32)
    d_data = cuda.to_device(data.astype(np.float32))
    d_starts = cuda.to_device(starts)
    d_results = cuda.device_array((len(starts), block_size), dtype=np.float32)

    # Configure the kernel
    threads_per_block = num_cores
    blocks_per_grid = (len(starts) + threads_per_block - 1) // threads_per_block

    # Execute the kernel
    process_blocks[blocks_per_grid, threads_per_block](d_data, d_starts, block_size, d_results)

    # Copy results back from GPU
    cuda.synchronize()
    fft_results_real = d_results.copy_to_host()

    # Perform FFT on the GPU using scikit-cuda
    fft_results = np.zeros((len(starts), block_size // 2 + 1), dtype=np.complex64)
    plan = cu_fft.Plan(block_size, np.float32, np.complex64)

    for i in range(len(starts)):
        d_block = gpuarray.to_gpu(fft_results_real[i])
        d_fft_result = gpuarray.empty(block_size // 2 + 1, np.complex64)
        cu_fft.fft(d_block, d_fft_result, plan)
        fft_results[i] = d_fft_result.get()

    # Compute the mean amplitude for each frequency
    mean_amplitudes = np.mean(np.abs(fft_results), axis=0)

    # Find and print the frequencies with mean amplitude above the threshold
    frequencies = np.fft.rfftfreq(block_size, d=1/sample_rate)
    for freq, amp in zip(frequencies, mean_amplitudes):
        if amp > threshold:
            print(f"Frequency: {freq:.2f} Hz, Mean Amplitude: {amp:.2f}")


def compare_times_gpu(wav_path, block_size, offset, threshold, num_gpus):
    # Measure time for the original function
    start_time = time.time()
    sequential(wav_path, block_size, offset, threshold)
    original_time = time.time() - start_time

    # Measure time for the parallel function
    start_time = time.time()
    parallel_gpu(wav_path, block_size, offset, threshold, nump_gpus)
    parallel_time = time.time() - start_time

    # Calculate speedup
    speedup = original_time / parallel_time

    print(f"Original function time: {original_time:.4f} seconds")
    print(f"Parallel function time: {parallel_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    #generate_wav_file('test_signal.wav', frequency=440, amplitude=0.5, duration=600.0)
    parallel_gpu('nicht_zu_laut_abspielen.wav', block_size=256, offset=10, threshold=4000, num_cores=20)
    #compare_times('test_signal.wav', block_size=128, offset=10, threshold=4000, num_cores=16)
