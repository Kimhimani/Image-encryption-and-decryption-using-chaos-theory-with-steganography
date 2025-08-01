import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt


# === Load Images ===
secret_img = cv2.imread('secret.png', cv2.IMREAD_GRAYSCALE)
if secret_img is None:
    raise ValueError("Secret image not found.")


cover_img = cv2.imread('cover.png', cv2.IMREAD_GRAYSCALE)
if cover_img is None:
    raise ValueError("Cover image not found.")


# === Resize ===
secret_img = cv2.resize(secret_img, (256, 256))
cover_img = cv2.resize(cover_img, (256, 256))


# === DWT Transformations ===
secret_float = np.float32(secret_img)
cover_float = np.float32(cover_img)


# Apply 2D DWT using Daubechies wavelet
wavelet = 'db8'  # Daubechies 8 wavelet


# Extract subbands: (LL, (LH, HL, HH))
secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, wavelet)
cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, wavelet)


# === CORRECTED EMBEDDING ===
embedded_LL = cover_LL.copy()        # Keep cover LL unchanged
embedded_LH = cover_LH.copy()        # Keep cover LH unchanged
embedded_HL = secret_LL / 10         # REPLACE cover HL with secret LL
embedded_HH = secret_LH / 10         # REPLACE cover HH with secret LH


# Reconstruct embedded coefficients
embedded_coeffs = (embedded_LL, (embedded_LH, embedded_HL, embedded_HH))


# IDWT: Get Watermarked Image
watermarked_img = pywt.idwt2(embedded_coeffs, wavelet)


# === CORRECTED EXTRACTION PROCESS ===
watermarked_coeffs = pywt.dwt2(watermarked_img, wavelet)
watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = watermarked_coeffs


# CORRECTED EXTRACTION:
extracted_secret_LL = watermarked_HL * 10    # Extract secret LL from HL
extracted_secret_LH = watermarked_HH * 10    # Extract secret LH from HH
extracted_secret_HL = np.zeros_like(secret_HL)  # Not embedded, set to zero
extracted_secret_HH = np.zeros_like(secret_HH)  # Not embedded, set to zero


# Reconstruct extracted secret
extracted_secret_coeffs = (extracted_secret_LL, (extracted_secret_LH, extracted_secret_HL, extracted_secret_HH))
extracted_secret = pywt.idwt2(extracted_secret_coeffs, wavelet)




# === Display Output ===
fig, axs = plt.subplots(2, 2, figsize=(18, 10))


axs[0, 0].imshow(cover_img, cmap='gray')
axs[0, 0].set_title("Original Cover Image")
axs[0, 0].axis('off')


axs[0, 1].imshow(secret_img, cmap='gray')
axs[0, 1].set_title("Original Secret Image")
axs[0, 1].axis('off')


axs[1, 0].imshow(np.clip(watermarked_img, 0, 255), cmap='gray')
axs[1, 0].set_title("Watermarked Image")
axs[1, 0].axis('off')


axs[1, 1].imshow(np.clip(extracted_secret, 0, 255), cmap='gray')
axs[1, 1].set_title("Extracted Secret Image")
axs[1, 1].axis('off')


plt.tight_layout()
plt.show()