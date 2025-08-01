import numpy as np
import cv2

# Create 32x32 image with horizontal stripes (2 pixels high)
secret = np.zeros((32, 32), dtype=np.uint8)

for row in range(0, 32, 2):
    color = 255 if (row // 2) % 2 == 0 else 0
    secret[row:row+2, :] = color

cv2.imwrite("secret2.png", secret)
print("Generated striped secret.png with size 32x32")
