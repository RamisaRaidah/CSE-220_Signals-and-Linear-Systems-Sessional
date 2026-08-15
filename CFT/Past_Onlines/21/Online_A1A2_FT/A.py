import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
image = plt.imread('noisy_image.png')  # Replace with your image file path
# show the image
plt.figure()
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.show()
if image.ndim == 3:
    image = np.mean(image, axis=2)  # Convert to grayscale
image = image / 255.0  # Normalize to range [0, 1]
print(image.shape)
sample_rate = 1000

# ---------------------------------------------------------------------
# Step 1: Apply the (Discrete) Fourier Transform row by row.
# Each row of the image is a 1D signal; taking its FFT tells us which
# column-frequencies are present in that row.
# ---------------------------------------------------------------------
F = np.fft.fft(image, axis=1)          # FFT along each row
magnitude = np.abs(F)

# ---------------------------------------------------------------------
# Step 2: Figure out which frequencies are the noise.
# The secret letter is a smooth, low-frequency shape, so it only
# contributes strongly to the DC term and the very first frequency bin.
# The periodic vertical-stripe noise, on the other hand, shows up as a
# few sharp spikes sitting well above the rest of the spectrum.
# We find those spikes automatically by looking for bins (other than
# the DC bin) whose average magnitude across all rows is unusually
# high (mean + 2*std of the non-DC spectrum).
# ---------------------------------------------------------------------
avg_magnitude = np.mean(magnitude, axis=0)
avg_magnitude_no_dc = avg_magnitude.copy()
avg_magnitude_no_dc[0] = 0  # ignore the DC term when searching for noise

threshold = avg_magnitude_no_dc.mean() + 2 * avg_magnitude_no_dc.std()
noise_freqs = np.where(avg_magnitude_no_dc > threshold)[0]
print("Detected noise frequency bins:", noise_freqs)

# ---------------------------------------------------------------------
# Step 3: Filter out the noise frequencies (notch filter) and invert.
# ---------------------------------------------------------------------
F_filtered = F.copy()
F_filtered[:, noise_freqs] = 0

denoised_image = np.real(np.fft.ifft(F_filtered, axis=1))

# Rescale to [0, 1] so the recovered letter is easy to see
denoised_image = denoised_image - denoised_image.min()
denoised_image = denoised_image / denoised_image.max()
denoised_image = denoised_image * 255.0

plt.imsave('denoised_image.png', denoised_image, cmap='gray')
plt.figure()
plt.title('Denoised Image')
plt.imshow(denoised_image, cmap='gray')
plt.show()