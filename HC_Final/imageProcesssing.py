import cv2
import cupy as cp
import cupyx.scipy.ndimage as ndi
from cupyx.profiler import benchmark
import numpy as np
from scipy import ndimage
import time
import pandas as pd



def load_and_preprocess_image(image_path, filter_choice, scale_factor=1):
    image = cv2.imread(image_path)

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions based on scale factor
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    if filter_choice == "gaussian_blur" or filter_choice == "sharpen":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Optional for these filters
        preprocessed_image = cv2.resize(gray_image, (new_width, new_height))

    elif filter_choice == "edge_detection":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Required for edge detection
        preprocessed_image = cv2.resize(gray_image, (new_width, new_height))

    elif filter_choice == "nlm_denoising":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_image = cv2.resize(gray_image, (new_width, new_height))

    return preprocessed_image

# Transfer image data from CPU to GPU (np to cp)
def transfer_to_gpu(image):
    gpu_image = cp.asarray(image)
    return gpu_image

# Transfer processed image data from GPU to CPU (cp to np)
def transfer_to_cpu(gpu_image):
    cpu_image = cp.asnumpy(gpu_image)
    return cpu_image

def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    """Create a Gaussian kernel."""
    kernel_1d = cp.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    gaussian_kernel = cp.exp(-(kernel_1d ** 2) / (2. * sigma ** 2))
    gaussian_kernel /= cp.sum(gaussian_kernel)
    return gaussian_kernel

def apply_gaussian_blur_gpu(gpu_image, kernel_size=5, sigma=1.0, gaussian_kernel=None):
    # Create the Gaussian kernel if not provided
    if gaussian_kernel is None:
        gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

    # Apply the Gaussian filter using cupy.ndimage.convolve
    gpu_image_blurred_x = ndi.convolve(gpu_image, gaussian_kernel[None, :], mode='mirror')
    gpu_image_blurred = ndi.convolve(gpu_image_blurred_x, gaussian_kernel[:, None], mode='mirror')

    return gpu_image_blurred

# Sobel Edge Detection using CuPy
def apply_sobel_edge_detection_gpu(gpu_image):
    # Sobel kernels for x and y directions
    sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
    sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)

    # Convolve the image with Sobel kernels using cupyx.scipy.ndimage
    gradient_x = ndi.convolve(gpu_image, sobel_x)
    gradient_y = ndi.convolve(gpu_image, sobel_y)

    # Compute gradient magnitude
    magnitude = cp.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Normalize the result to range [0, 255]
    magnitude = (magnitude / magnitude.max()) * 255

    # Apply threshold to focus on stronger edges
    threshold_value = 200  # Tune this value as needed
    magnitude = cp.where(magnitude > threshold_value, magnitude, 0)

    return magnitude.astype(cp.uint8)

# Sharpening Filters using Cupy
def apply_sharpening_gpu(original_gpu_image, blurred_gpu_image, alpha=2.0):
    # Convert image to float32 for better precision during operations
    original_gpu_image = original_gpu_image.astype(cp.float32)
    blurred_gpu_image = blurred_gpu_image.astype(cp.float32)

    # Calculate the sharpened image using the unsharp masking formula
    sharpened_image = original_gpu_image + alpha * (original_gpu_image - blurred_gpu_image)

    # Clip values to ensure they remain within valid range [0, 255]
    sharpened_image = cp.clip(sharpened_image, 0, 255)

    return sharpened_image.astype(cp.uint8)

# Gaussian Blur without CuPy (Numpy)
def apply_gaussian_blur_cpu(image, kernel_size=5, sigma=1.0):
    # Create a Gaussian kernel using NumPy
    kernel_1d = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    gaussian_kernel = np.exp(-(kernel_1d ** 2) / (2. * sigma ** 2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    # Apply the Gaussian filter along the x and y axes (separable filter)
    image_blurred_x = ndimage.convolve1d(image, gaussian_kernel, axis=0, mode='reflect')
    image_blurred = ndimage.convolve1d(image_blurred_x, gaussian_kernel, axis=1, mode='reflect')

    return image_blurred

# Sobel Edge Detection without CuPy (Numpy)
def apply_sobel_edge_detection_cpu(image):
    # Sobel kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Convolve the image with Sobel kernels using scipy.ndimage
    gradient_x = ndimage.convolve(image, sobel_x)
    gradient_y = ndimage.convolve(image, sobel_y)

    # Compute gradient magnitude
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Normalize the result to range [0, 255]
    magnitude = (magnitude / magnitude.max()) * 255

    # Apply threshold to focus on stronger edges
    threshold_value = 200  # Tune this value as needed
    magnitude = np.where(magnitude > threshold_value, magnitude, 0)

    return magnitude.astype(np.uint8)

# Sharpening Filters without CuPy (Numpy)
def apply_sharpening_cpu(original_image, blurred_image, alpha=2.0):
    # Convert image to float32 for better precision during operations
    original_image = original_image.astype(np.float32)
    blurred_image = blurred_image.astype(np.float32)

    # Calculate the sharpened image using the unsharp masking formula
    sharpened_image = original_image + alpha * (original_image - blurred_image)

    # Clip values to ensure they remain within valid range [0, 255]
    sharpened_image = np.clip(sharpened_image, 0, 255)

    return sharpened_image.astype(np.uint8)

# Non-local means denoising using CuPy
def apply_nlm_denoising_gpu(gpu_image, patch_size=2, h=10):
    """
    Applies Non-Local Means Denoising on the GPU with batch processing.
    :param gpu_image: CuPy image array on the GPU.
    :param patch_size: Size of the patch used for denoising.
    :param h: Filtering parameter, controlling smoothing strength.
    :return: Denoised image on the GPU.
    """
    patch_radius = patch_size // 2

    # Pad the image to handle border cases
    padded_image = cp.pad(gpu_image, ((patch_radius, patch_radius), (patch_radius, patch_radius)), mode='constant', constant_values=0)

    denoised_image = cp.zeros_like(gpu_image, dtype=cp.float32)  # Ensure float32 type
    weights = cp.zeros_like(denoised_image, dtype=cp.float32)  # Ensure float32 type

    # Loop over the image using a single loop for batch processing
    for i in range(patch_radius, padded_image.shape[0] - patch_radius):
        for j in range(patch_radius, padded_image.shape[1] - patch_radius):
            # Extract the reference patch
            reference_patch = padded_image[i - patch_radius:i + patch_radius + 1, j - patch_radius:j + patch_radius + 1]

            # Iterate over neighboring pixels within the patch
            for m in range(-patch_radius, patch_radius + 1):
                for n in range(-patch_radius, patch_radius + 1):
                    neighbor_patch = padded_image[i + m - patch_radius:i + m + patch_radius + 1, j + n - patch_radius:j + n + patch_radius + 1]

                    # Ensure that the neighbor patch has the correct shape
                    if neighbor_patch.shape == reference_patch.shape:
                        # Calculate the distance (similarity) between patches
                        distance = cp.sum((reference_patch - neighbor_patch) ** 2)

                        # Compute the weight based on the patch similarity
                        weight = cp.exp(-distance / (h ** 2))

                        # Weighted sum update
                        denoised_image[i - patch_radius, j - patch_radius] += weight * padded_image[i + m, j + n]
                        weights[i - patch_radius, j - patch_radius] += weight

    # Normalize the result by weights
    denoised_image = cp.divide(denoised_image, cp.maximum(weights, 1e-6))  # Avoid division by zero

    # Convert denoised_image to uint8 type
    return cp.clip(denoised_image, 0, 255).astype(cp.uint8)  # Clip values to 0-255 and convert to uint8

# Non-local means denoising without Cupy (Numpy)
def apply_nlm_denoising_cpu(image, patch_size=2, h=10):
    """
    Applies Non-Local Means Denoising on the CPU.
    :param image: NumPy image array on the CPU.
    :param patch_size: Size of the patch used for denoising.
    :param h: Filtering parameter, controlling smoothing strength.
    :return: Denoised image on the CPU.
    """
    patch_radius = patch_size // 2

    # Pad the image to handle border cases
    padded_image = np.pad(image, ((patch_radius, patch_radius), (patch_radius, patch_radius)), mode='constant', constant_values=0)

    denoised_image = np.zeros_like(image)

    for i in range(patch_radius, padded_image.shape[0] - patch_radius):
        for j in range(patch_radius, padded_image.shape[1] - patch_radius):
            # Extract the reference patch
            reference_patch = padded_image[i - patch_radius:i + patch_radius + 1, j - patch_radius:j + patch_radius + 1]

            # Compute the weighted sum over all patches within the window
            weighted_sum = 0.0
            total_weight = 0.0

            # Iterate over neighboring pixels within the patch
            for m in range(-patch_radius, patch_radius + 1):
                for n in range(-patch_radius, patch_radius + 1):
                    neighbor_patch = padded_image[i + m - patch_radius:i + m + patch_radius + 1, j + n - patch_radius:j + n + patch_radius + 1]

                    # Ensure that the neighbor patch has the correct shape
                    if neighbor_patch.shape == reference_patch.shape:
                        # Calculate the distance (similarity) between patches
                        distance = np.sum((reference_patch - neighbor_patch) ** 2)

                        # Compute the weight based on the patch similarity
                        weight = np.exp(-distance / (h ** 2))

                        # Weighted sum
                        weighted_sum += weight * padded_image[i + m, j + n]
                        total_weight += weight

            # Normalize the result
            if total_weight > 0:
                denoised_image[i - patch_radius, j - patch_radius] = weighted_sum / total_weight
            else:
                denoised_image[i - patch_radius, j - patch_radius] = padded_image[i, j]  # Use original value if no weights

    return denoised_image.astype(np.uint8)

def main():
    image_path = "1.jpg"

    filter_choices = ["nlm_denoising"]
    scale_factors = [0.125]

    #filter_choices = ["gaussian_blur", "edge_detection", "sharpen"]
    #scale_factors = [0.125]

    for filter_choice in filter_choices:
        # Initialize an empty DataFrame to store CPU and GPU times for each scale factor
        df = pd.DataFrame(columns=['Scale Factor', 'GPU Approach', 'CPU Approach'])

        for scale_factor in scale_factors:
            # 1. Load and preprocess the image on the CPU
            preprocessed_image = load_and_preprocess_image(image_path, filter_choice, scale_factor = scale_factor)

             ### GPU Warm-up ###
            # Transfer the preprocessed image to the GPU
            gpu_image = transfer_to_gpu(preprocessed_image)

            # Apply a dummy filter on the GPU to "warm up"
            if filter_choice == "gaussian_blur":
                apply_gaussian_blur_gpu(gpu_image)
            elif filter_choice == "edge_detection":
                apply_sobel_edge_detection_gpu(gpu_image)
            elif filter_choice == "sharpen":
                gpu_image_blur = apply_gaussian_blur_gpu(gpu_image)
                apply_sharpening_gpu(gpu_image, gpu_image_blur)


            ### GPU Processing ###
            # Transfer the preprocessed image to the GPU
            gpu_image = transfer_to_gpu(preprocessed_image)

            # Time the GPU processing
            start_gpu_time = time.time()

            # Apply a filter on the GPU
            if filter_choice == "gaussian_blur":
                gpu_image_processed = apply_gaussian_blur_gpu(gpu_image)
            elif filter_choice == "edge_detection":
                gpu_image = apply_gaussian_blur_gpu(gpu_image)
                gpu_image_processed = apply_sobel_edge_detection_gpu(gpu_image)
            elif filter_choice == "sharpen":
                gpu_image_blur = apply_gaussian_blur_gpu(gpu_image)
                gpu_image_processed = apply_sharpening_gpu(gpu_image, gpu_image_blur)
            elif filter_choice == "nlm_denoising":
                gpu_image_blur = apply_gaussian_blur_gpu(gpu_image)
                gpu_image_processed = apply_nlm_denoising_gpu(gpu_image_blur)

            # Measure the elapsed time for GPU processing
            gpu_elapsed_time = time.time() - start_gpu_time

            # Free unused memory blocks after GPU processing
            cp.get_default_memory_pool().free_all_blocks()

            # Transfer the processed image back to the CPU
            processed_image_gpu = transfer_to_cpu(gpu_image_processed)

            ### CPU Processing ###
            # Time the CPU processing
            start_cpu_time = time.time()

            # Apply the same filter on the CPU
            if filter_choice == "gaussian_blur":
                cpu_image_processed = apply_gaussian_blur_cpu(preprocessed_image)
            elif filter_choice == "edge_detection":
                preprocessed_image = apply_gaussian_blur_cpu(preprocessed_image)
                cpu_image_processed = apply_sobel_edge_detection_cpu(preprocessed_image)
            elif filter_choice == "sharpen":
                cpu_image_blur = apply_gaussian_blur_cpu(preprocessed_image)
                cpu_image_processed = apply_sharpening_cpu(preprocessed_image, cpu_image_blur)
            elif filter_choice == "nlm_denoising":
                cpu_image_blur = apply_gaussian_blur_cpu(preprocessed_image)
                cpu_image_processed = apply_nlm_denoising_cpu(cpu_image_blur)

            # Measure the elapsed time for CPU processing
            cpu_elapsed_time = time.time() - start_cpu_time

            # Save the processed images with a naming convention based on the filter and scale factor
            gpu_image_filename = f"results/{filter_choice}_gpu_sf{scale_factor}.jpg"
            cpu_image_filename = f"results/{filter_choice}_cpu_sf{scale_factor}.jpg"
            cv2.imwrite(gpu_image_filename, processed_image_gpu)
            cv2.imwrite(cpu_image_filename, cpu_image_processed)

            # Append the performance data to the DataFrame using pd.concat
            new_row = pd.DataFrame({
                'Scale Factor': [scale_factor],
                'GPU Approach': [gpu_elapsed_time],
                'CPU Approach': [cpu_elapsed_time]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        # Save the DataFrame to an Excel file named after the filter choice
        excel_filename = f"excel/{filter_choice}_performance.xlsx"
        df.to_excel(excel_filename, index=False)

if __name__ == "__main__":
    main()
