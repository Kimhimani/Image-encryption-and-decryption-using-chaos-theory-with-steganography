import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
import os
import base64

# === AES-Based Key Derivation ===
def derive_aes_key(passphrase: str, salt: bytes, length: int = 32) -> str:
    #Derive a secure key using PBKDF2 (can be used as a seed for chaos maps).
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(passphrase.encode())
    return key.hex()  # Return as hex string to use in SHA256

# === Chaos Parameter Generator ===
def generate_chaos_params(secret_key):
    key_hash = hashlib.sha256(secret_key.encode()).hexdigest()
    x0 = int(key_hash[0:8], 16) % 1000 / 1000.0
    r  = 3.57 + (int(key_hash[8:16], 16) % 43) / 100.0#([3.57- 4.0] is the standard or highly chaotic value)
    a  = 1.4 + (int(key_hash[16:24], 16) % 10) / 100.0#(must be close to 1.4 so that it can work up to it's full capacity)
    b  = 0.3 + (int(key_hash[24:32], 16) % 10) / 100.0#(must be aronund 0.3 for it to be ideal for chaos)
    return x0, r, a, b

# === Henon Map ===
def henon_map(a, b, size):
    x = np.zeros(size)
    y = np.zeros(size)
    x[0], y[0] = 0.1, 0.1
    for i in range(1, size):
        # Prevent overflow by clamping the value
        try:
            x_sq = x[i - 1] ** 2
        except OverflowError:
            x_sq = 1e10
        x_sq = np.clip(x_sq, 0, 1e10)  # Limit to avoid overflow
        x[i] = 1 - a * x_sq + y[i - 1]
        y[i] = b * x[i - 1]
    return x, y



# === Logistic Map ===
def logistic_map(x0, r, size):
    x = np.zeros(size)
    x[0] = x0
    for i in range(1, size):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x

# === Main Program ===
img = cv2.imread('secret2.png', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
size = h * w

# Prompt user for passphrase
passphrase = input("Enter passphrase to generate AES-based key: ")
salt = b'secure_salt_1234'  # Salt ensures that even the same passphrase results in different keys.
aes_key = derive_aes_key(passphrase, salt)

# Generate chaos parameters
x0, r, a, b = generate_chaos_params(aes_key)

print(f"Using params -> a: {a}, b: {b}, x0: {x0}, r: {r}")


# Henon Map
hx, hy = henon_map(a, b, size)
henon_seq = np.abs(hx + hy) % 1
scramble_order = np.argsort(henon_seq)

# Logistic Map
log_seq = logistic_map(x0, r, size)
log_scaled = np.floor(log_seq * 256).astype(np.uint8)

# Encryption
flat_img = img.flatten()
scrambled = flat_img[scramble_order]
encrypted = np.bitwise_xor(scrambled, log_scaled)
cv2.imwrite("key_based_encrypted.png", encrypted.reshape(h, w))

# Decryption
de_scrambled = np.bitwise_xor(encrypted, log_scaled)
unscrambled = np.zeros_like(de_scrambled)
unscrambled[scramble_order] = de_scrambled
decrypted_img = unscrambled.reshape(h, w)

# Show Results
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(132), plt.imshow(encrypted.reshape(h, w), cmap='gray'), plt.title("Encrypted")
plt.subplot(133), plt.imshow(decrypted_img, cmap='gray'), plt.title("Decrypted")
plt.show()
