import cv2
import numpy as np
import math


def fft(x):
    """
    Compute 1D FFT (recursive radix-2 Cooley-Tukey). N must be a power of two.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("fft: length must be a power of two, got %d" % N)
    if N == 1:
        return x

    even = fft(x[0::2])
    odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + factor * odd, even - factor * odd])


def ifft(X):
    """
    Compute 1D inverse FFT using the FFT function (conjugate-transform-conjugate trick).
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)
    x = fft(np.conjugate(X))
    return np.conjugate(x) / N


def find_row_shift(orig_row, shifted_row):
    """
    Find how much shifted_row = roll(orig_row, m) using FFT cross-correlation.
    Returns m, the amount shifted_row was circularly shifted to the right.
    """
    X = fft(orig_row.astype(complex))
    Y = fft(shifted_row.astype(complex))

    # Cross-power spectrum: conj(X) * Y  ->  peak of its inverse FFT sits at lag m
    C = np.conjugate(X) * Y
    r = ifft(C).real

    return int(np.argmax(r))


def reconstruct_image_using_fft(original_path, shifted_path, output_path):

    original_img = cv2.imread(original_path)
    shifted_img = cv2.imread(shifted_path)

    if original_img is None or shifted_img is None:
        print("Error: Could not load images.")
        return

    if original_img.shape != shifted_img.shape:
        print("Error: Image dimensions do not match.")
        return

    # Convert the original and shifted color images to grayscale.
    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    shift_gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY)

    H, W = shift_gray.shape
    reconstructed_img = np.zeros_like(shifted_img)

    print("Reconstructing image using manual FFT...")

    for row in range(H):
        orig_row = orig_gray[row, :]
        shifted_row = shift_gray[row, :]

        shift_amount = find_row_shift(orig_row, shifted_row)

        # shifted_row = roll(orig_row, shift_amount), so undo by rolling
        # back the other way, per color channel.
        for c in range(3):
            reconstructed_img[row, :, c] = np.roll(shifted_img[row, :, c], -shift_amount)

    recon_gray = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2GRAY)
    diff = np.abs(orig_gray.astype(int) - recon_gray.astype(int))
    print("Max abs difference after reconstruction :", diff.max())
    print("Mean abs difference after reconstruction:", diff.mean())

    cv2.imwrite(output_path, reconstructed_img)


if __name__ == "__main__":
    reconstruct_image_using_fft("original_image.png", "shifted_image.jpg", "reconstructed_image_fft.jpg")