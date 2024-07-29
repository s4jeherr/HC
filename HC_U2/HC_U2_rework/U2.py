import numpy as np
import cupy as cp
from scipy.io import wavfile
from scipy.io.wavfile import write
from pydub import AudioSegment
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def generate_wav_file(filename, frequency, amplitude, duration, sample_rate=44100):
    """
    Generate a WAV file with a sine wave signal.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    signal = np.clip(signal, -1, 1)
    pcm_signal = np.int16(signal * 32767)
    write(filename, sample_rate, pcm_signal)

def split_audio(audio_file, block_size, offset):
    """
    Splits an audio file into blocks of size `block_size` with an offset `offset`.
    """
    audio = AudioSegment.from_file(audio_file)
    blocks = []
    total_duration = len(audio)
    current_position = 0

    while current_position < total_duration:
        block = audio[current_position:current_position + block_size]
        blocks.append(block)
        current_position += offset

    return blocks

def audio_segment_to_array(segment):
    """
    Convert an AudioSegment to a NumPy array.
    """
    samples = np.array(segment.get_array_of_samples())
    return samples.reshape(-1, segment.channels)

def perform_fft_on_block(block, target_length):
    """
    Perform FFT on a given audio block on the CPU.
    """
    audio_array = audio_segment_to_array(block)
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    # Ensure the audio block has the target length
    if len(audio_array) < target_length:
        padded_array = np.pad(audio_array, (0, target_length - len(audio_array)), 'constant')
    else:
        padded_array = audio_array[:target_length]

    fft_result = np.fft.fft(padded_array)
    return fft_result

def sequential(wav_path, block_size, offset, threshold):
    if not (64 <= block_size <= 512):
        raise ValueError("Block size must be between 64 and 512")

    if offset >= block_size:
        raise ValueError("Offset must be smaller than the block size")

    sample_rate, data = wavfile.read(wav_path)

    blocks = split_audio(wav_path, block_size, offset)
    fft_results = [perform_fft_on_block(block, block_size) for block in blocks]

    fft_results = np.array(fft_results)
    #print(f"FFT results shape: {fft_results.shape}, dtype: {fft_results.dtype}")

    mean_amplitudes = np.mean(np.abs(fft_results), axis=0)
    frequencies = np.fft.fftfreq(block_size, d=1/sample_rate)

    #print(f"Frequencies: {frequencies}")
    #print(f"Mean amplitudes: {mean_amplitudes}")

    #for freq, amp in zip(frequencies, mean_amplitudes):
        #if amp > threshold:
            #print(f"Frequency: {freq:.2f} Hz, Mean Amplitude: {amp:.2f}")

def perform_fft_on_block_parallel(block, target_length):
    return perform_fft_on_block(block, target_length)

def parallel_cpu(wav_path, block_size, offset, threshold, num_cores):
    if not (64 <= block_size <= 512):
        raise ValueError("Block size must be between 64 and 512")

    if offset >= block_size:
        raise ValueError("Offset must be smaller than the block size")

    sample_rate, data = wavfile.read(wav_path)

    blocks = split_audio(wav_path, block_size, offset)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        fft_results = list(executor.map(perform_fft_on_block_parallel, blocks, [block_size] * len(blocks)))

    fft_results = np.array(fft_results)
    #print(f"FFT results shape: {fft_results.shape}, dtype: {fft_results.dtype}")

    mean_amplitudes = np.mean(np.abs(fft_results), axis=0)
    frequencies = np.fft.fftfreq(block_size, d=1/sample_rate)

    #print(f"Frequencies: {frequencies}")
    #print(f"Mean amplitudes: {mean_amplitudes}")

    #for freq, amp in zip(frequencies, mean_amplitudes):
        #if amp > threshold:
            #print(f"Frequency: {freq:.2f} Hz, Mean Amplitude: {amp:.2f}")


def audio_segment_to_array(segment):
    """
    Convert an AudioSegment to a NumPy array.
    """
    samples = np.array(segment.get_array_of_samples())
    return samples.reshape(-1, segment.channels)

def perform_fft_on_stream(block, stream, target_length):
    """
    Perform FFT on a given audio block using CuPy within a specified CUDA stream.
    """
    audio_array = audio_segment_to_array(block)
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    # Ensure the audio block has the target length
    if len(audio_array) < target_length:
        padded_array = np.pad(audio_array, (0, target_length - len(audio_array)), 'constant')
    else:
        padded_array = audio_array[:target_length]

    with stream:
        gpu_audio_array = cp.asarray(padded_array, dtype=cp.float32)
        fft_result = cp.fft.fft(gpu_audio_array)

    return fft_result

def process_blocks_on_gpu(blocks, num_cores, block_size):
    """
    Process audio blocks using multiple CUDA streams to approximate using multiple GPU cores.
    """
    streams = [cp.cuda.Stream() for _ in range(num_cores)]
    fft_results = []

    for i, block in enumerate(blocks):
        stream = streams[i % num_cores]
        fft_result = perform_fft_on_stream(block, stream, block_size)
        fft_results.append(fft_result)

    # Ensure all streams are synchronized
    for stream in streams:
        stream.synchronize()

    # Ensure fft_results is not empty and not a 0-d array
    fft_results = [result for result in fft_results if result is not None]
    if fft_results and isinstance(fft_results[0], cp.ndarray) and fft_results[0].ndim > 0:
        # Stack fft_results to 2D array
        fft_results = cp.stack(fft_results)
        #print(f"FFT results reshaped to 2D array: {fft_results.shape}, dtype: {fft_results.dtype}")
        return fft_results
    else:
        return cp.array([])

def parallel_gpu(wav_path, block_size, offset, threshold, num_cores):
    if not (64 <= block_size <= 512):
        raise ValueError("Block size must be between 64 and 512")

    if offset >= block_size:
        raise ValueError("Offset must be smaller than the block size")

    sample_rate, data = wavfile.read(wav_path)

    blocks = split_audio(wav_path, block_size, offset)
    fft_results = process_blocks_on_gpu(blocks, num_cores, block_size)

    #print(f"FFT results shape: {fft_results.shape}, dtype: {fft_results.dtype}")

    if fft_results.size == 0:
        print("No FFT results to process.")
        return

    # Calculate mean amplitudes across blocks for each frequency
    mean_amplitudes = cp.mean(cp.abs(fft_results), axis=0)
    #print(f"Mean amplitudes shape: {mean_amplitudes.shape}, dtype: {mean_amplitudes.dtype}")

    frequencies = cp.fft.rfftfreq(block_size, d=1/sample_rate)
    frequencies = cp.asnumpy(frequencies)
    mean_amplitudes = cp.asnumpy(mean_amplitudes)

    #print(f"Frequencies: {frequencies}")
    #print(f"Mean amplitudes: {mean_amplitudes}")

    #for freq, amp in zip(frequencies, mean_amplitudes):
        #if amp > threshold:
            #print(f"Frequency: {freq:.2f} Hz, Mean Amplitude: {amp:.2f}")

def experiments_block_size():

    """
    Testing different Block sizes
    """

    parameters = [0, 1, 100, 4, 16]
    block_sizes = [64, 128, 256, 512]
    df = pd.DataFrame(columns=['Sequential', "CPU", "GPU"], index=block_sizes)

    generate_wav_file('test_signal1.wav', frequency=440, amplitude=0.5, duration=30)

    for block_size in block_sizes:
        start_time1 = time.time()
        sequential('test_signal1.wav', block_size, parameters[1], parameters[2])
        run_time1 = time.time() - start_time1

        start_time2 = time.time()
        parallel_cpu('test_signal1.wav', block_size, parameters[1], parameters[2], parameters[3])
        run_time2 = time.time() - start_time2

        start_time3 = time.time()
        parallel_gpu('test_signal1.wav', block_size, parameters[1], parameters[2], parameters[4])
        run_time3 = time.time() - start_time3

        df.loc[block_size] = pd.Series({'Sequential':round(run_time1,1), 'CPU':round(run_time2,1),'GPU':round(run_time3,1)})

    df.to_excel("block_size.xlsx")

def experiments_offset():

    """
    Testing different offsets
    """

    parameters = [256, 0, 100, 4, 16]
    offsets = [1, 64, 128, 196]
    df = pd.DataFrame(columns=['Sequential', "CPU", "GPU"], index=offsets)

    generate_wav_file('test_signal1.wav', frequency=440, amplitude=0.5, duration=300)

    for offset in offsets:
        start_time1 = time.time()
        sequential('test_signal1.wav', parameters[0], offset, parameters[2])
        run_time1 = time.time() - start_time1

        start_time2 = time.time()
        parallel_cpu('test_signal1.wav', parameters[0], offset, parameters[2], parameters[3])
        run_time2 = time.time() - start_time2

        start_time3 = time.time()
        parallel_gpu('test_signal1.wav', parameters[0], offset, parameters[2], parameters[4])
        run_time3 = time.time() - start_time3

        df.loc[offset] = pd.Series({'Sequential':round(run_time1,1), 'CPU':round(run_time2,1),'GPU':round(run_time3,1)})

    df.to_excel("offsets.xlsx")

def experiments_duration():

    """
    Testing different durations
    """

    parameters = [256, 1, 100, 4, 16]
    durations = [10, 30, 120, 300]
    df = pd.DataFrame(columns=['Sequential', "CPU", "GPU"], index=durations)

    for dur in durations:
        generate_wav_file('test_signal1.wav', frequency=440, amplitude=0.5, duration=dur)

        start_time1 = time.time()
        sequential('test_signal1.wav', parameters[0], parameters[1], parameters[2])
        run_time1 = time.time() - start_time1

        start_time2 = time.time()
        parallel_cpu('test_signal1.wav', parameters[0], parameters[1], parameters[2], parameters[3])
        run_time2 = time.time() - start_time2

        start_time3 = time.time()
        parallel_gpu('test_signal1.wav', parameters[0], parameters[1], parameters[2], parameters[4])
        run_time3 = time.time() - start_time3

        df.loc[dur] = pd.Series({'Sequential':round(run_time1,1), 'CPU':round(run_time2,1),'GPU':round(run_time3,1)})

    df.to_excel("durations.xlsx")

def experiments_cores():

    """
    Testing different core numbers
    """

    parameters = [256, 1, 100, 0, 0]
    cpu_cores = [1, 2, 4, 8, 16]
    gpu_cores = [1, 16, 32, 64, 128]
    index_labels = [f"{cpu}/{gpu}" for cpu, gpu in zip(cpu_cores, gpu_cores)]
    df = pd.DataFrame(columns=['Sequential', "CPU", "GPU"], index=index_labels)

    generate_wav_file('test_signal1.wav', frequency=440, amplitude=0.5, duration=30)

    for cpu_core, gpu_core in zip(cpu_cores, gpu_cores):

        start_time1 = time.time()
        sequential('test_signal1.wav', parameters[0], parameters[1], parameters[2])
        run_time1 = time.time() - start_time1

        start_time2 = time.time()
        parallel_cpu('test_signal1.wav', parameters[0], parameters[1], parameters[2], cpu_core)
        run_time2 = time.time() - start_time2

        start_time3 = time.time()
        parallel_gpu('test_signal1.wav', parameters[0], parameters[1], parameters[2], gpu_core)
        run_time3 = time.time() - start_time3

        df.loc[f"{cpu_core}/{gpu_core}"] = pd.Series({'Sequential':round(run_time1,1), 'CPU':round(run_time2,1),'GPU':round(run_time3,1)})

    df.to_excel("cores.xlsx")


if __name__ == '__main__':
    experiments_block_size()
    experiments_offset()
    experiments_duration()
    experiments_cores()

