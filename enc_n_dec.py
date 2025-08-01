import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes


# === AES Key Derivation ===
def derive_aes_key(passphrase: str, salt: bytes, length: int = 32) -> str:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(passphrase.encode())
    return key.hex()


# === Chaos Parameter Generator ===
def generate_chaos_params(secret_key):
    key_hash = hashlib.sha256(secret_key.encode()).hexdigest()
    x0 = int(key_hash[0:8], 16) % 1000 / 1000.0
    r  = 3.57 + (int(key_hash[8:16], 16) % 43) / 100.0
    a  = 1.4 + (int(key_hash[16:24], 16) % 10) / 100.0
    b  = 0.3 + (int(key_hash[24:32], 16) % 10) / 100.0
    return x0, r, a, b


# === Henon Map ===
def henon_map(a, b, size):
    x = np.zeros(size)
    y = np.zeros(size)
    x[0], y[0] = 0.1, 0.1
    for i in range(1, size):
        x_sq = min(x[i - 1] ** 2, 1e6)
        x[i] = 1 - a * x_sq + y[i - 1]
        y[i] = b * x[i - 1]
    seq = np.nan_to_num(x + y, nan=0.0, posinf=0.0, neginf=0.0)
    seq = np.abs(seq) % 1
    return seq


# === Stronger Logistic Map for XOR ===
def logistic_map(x0, r, size):
    x = np.zeros(size)
    x[0] = x0
    for i in range(1, size):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    # More randomness by modulating the scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.floor(np.mod(x * 1e6, 256)).astype(np.uint8)


# === Load grayscale image ===
def load_image_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image not found")
    if len(img.shape) == 2:
        return img  # Already grayscale
    elif len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format")


# === Encrypt ===
def encrypt_image(img, passphrase, salt=b'secure_salt_1234'):
    h, w = img.shape
    size = h * w


    key = derive_aes_key(passphrase, salt)
    x0, r, a, b = generate_chaos_params(key)


    # Henon map for scrambling
    henon_seq = henon_map(a, b, size)
    scramble_order = np.argsort(henon_seq)


    # Logistic map for XOR key
    xor_key = logistic_map(x0, r, size)


    flat_img = img.flatten()
    scrambled = flat_img[scramble_order]
    encrypted = np.bitwise_xor(scrambled, xor_key)


    return encrypted.reshape(h, w), scramble_order, xor_key


# === Decrypt ===
def decrypt_image(encrypted_img, scramble_order, xor_key):
    h, w = encrypted_img.shape
    size = h * w


    encrypted_flat = encrypted_img.flatten()
    de_scrambled = np.bitwise_xor(encrypted_flat, xor_key)


    # Invert the scramble
    inverse_order = np.argsort(scramble_order)
    original_flat = de_scrambled[inverse_order]


    return original_flat.reshape(h, w)


# === Main Program ===
if __name__ == "__main__":
    image_path = input("Enter path to image: ").strip()
    img = load_image_grayscale(image_path)


    passphrase = input("Enter passphrase: ").strip()
    encrypted_img, scramble_order, xor_key = encrypt_image(img, passphrase)


    decrypted_img = decrypt_image(encrypted_img, scramble_order, xor_key)


    # Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(132), plt.imshow(encrypted_img, cmap='gray'), plt.title("Encrypted")
    plt.subplot(133), plt.imshow(decrypted_img, cmap='gray'), plt.title("Decrypted")
    plt.tight_layout()
    plt.show()