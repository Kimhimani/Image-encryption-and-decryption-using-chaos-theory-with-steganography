import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Zigzag utility ===
def zigzag_indices(n):
    index_order = sorted(((x, y) for x in range(n) for y in range(n)),
                         key=lambda s: (s[0]+s[1], -s[1] if (s[0]+s[1])%2 else s[1]))
    return index_order

# === Load & Resize Images ===
secret_img = cv2.imread('secret2.png', cv2.IMREAD_GRAYSCALE)
cover_img = cv2.imread('cover.png', cv2.IMREAD_GRAYSCALE)
secret_img = cv2.resize(secret_img, (256, 256))
cover_img = cv2.resize(cover_img, (256, 256))

# === Convert to float and DCT ===
secret_dct = cv2.dct(np.float32(secret_img))
cover_dct = cv2.dct(np.float32(cover_img))

# === Zigzag setup ===
zz = zigzag_indices(256)
num_coeffs = 1024  # Tuneable

# === Prepare data for embedding ===
secret_zz_vals = [secret_dct[i, j] for i, j in zz[:num_coeffs]]

# Normalize to cover DCT high-frequency range
high_freq_positions = zz[-num_coeffs:]
cover_vals = [cover_dct[i, j] for i, j in high_freq_positions]

secret_norm = (secret_zz_vals - np.mean(secret_zz_vals)) / (np.std(secret_zz_vals) + 1e-5)
cover_mean = np.mean(cover_vals)
cover_std = np.std(cover_vals)
secret_scaled = secret_norm * cover_std + cover_mean

# === Embed secret into cover's high frequency
watermarked_dct = np.copy(cover_dct)
for (i, j), val in zip(high_freq_positions, secret_scaled):
    watermarked_dct[i, j] = val

# === Inverse DCT to get watermarked image
watermarked_img = cv2.idct(watermarked_dct)

# === Extract secret from watermarked DCT
extracted_vals = [watermarked_dct[i, j] for i, j in high_freq_positions]

# De-normalize back (optional, here just place directly)
recovered_dct = np.zeros_like(secret_dct)
for val, (i, j) in zip(extracted_vals, zz[:num_coeffs]):
    recovered_dct[i, j] = val  # restore into original zigzag positions

# Inverse DCT to recover the secret image
extracted_secret = cv2.idct(recovered_dct)

# === Visualization ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

axs[0, 0].imshow(cover_img, cmap='gray')
axs[0, 0].set_title("Original Cover Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(secret_img, cmap='gray')
axs[0, 1].set_title("Original Secret Image")
axs[0, 1].axis('off')

axs[0, 2].imshow(np.log(np.abs(secret_dct[:32, :32]) + 1), cmap='gray')
axs[0, 2].set_title("Secret DCT (Low-Freq, Log)")
axs[0, 2].axis('off')

axs[1, 0].imshow(np.clip(watermarked_img, 0, 255), cmap='gray')
axs[1, 0].set_title("Watermarked Image (Zigzag Accurate)")
axs[1, 0].axis('off')

axs[1, 1].imshow(np.clip(extracted_secret, 0, 255), cmap='gray')
axs[1, 1].set_title("Extracted Secret Image (Accurate)")
axs[1, 1].axis('off')

axs[1, 2].axis('off')
plt.tight_layout()
plt.show()
