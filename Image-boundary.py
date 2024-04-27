import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/mLc/Downloads/image_processing.png")

height, width, _ = img.shape

# Create an empty array to store the grayscale image
gray_img = np.zeros((height, width), dtype=np.uint8)

# Convert the image to grayscale using a for loop
for i in range(height):
    for j in range(width):
        # Compute the mean of the BGR channels for each pixel
        gray_img[i][j] = np.mean(img[i][j])

# Get the dimensions of the grayscale image
height, width = gray_img.shape

# Determine the size of the square
square_size = min(height, width)

# Calculate the cropping parameters
crop_x = (width - square_size) // 2
crop_y = (height - square_size) // 2

# Crop the image to make it square
cropped_img = gray_img[crop_y:crop_y+square_size, crop_x:crop_x+square_size]


# Read the cropped grayscale image
boundary_img = cropped_img

# Define boundary size
boundary_size = 20

# Get the dimensions of the boundary image
height, width = boundary_img.shape

# Determine the maximum dimension
max_dim = max(height, width)

# Create a square image with padding
square_img = np.zeros((max_dim + 2 * boundary_size, max_dim + 2 * boundary_size), dtype=np.uint8)

# Calculate padding for height and width
padding_height = (max_dim - height) // 2
padding_width = (max_dim - width) // 2

# Assign the content of boundary_img to the center of square_img with padding
square_img[boundary_size + padding_height: boundary_size + padding_height + height,
           boundary_size + padding_width: boundary_size + padding_width + width] = boundary_img

# Display the result with equal subplot sizes
plt.figure(figsize=(10, 5))  # Set the size of the figure
plt.subplot(1, 2, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(square_img, cmap='gray')
plt.title('After Adding Boundary')
plt.axis('off')
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()