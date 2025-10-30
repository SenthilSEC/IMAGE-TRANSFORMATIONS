# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import necessary libraries such as OpenCV, NumPy, and Matplotlib for image processing and visualization.

### Step2:
Read the input image using cv2.imread() and store it in a variable for further processing.

### Step3:
Apply various transformations like translation, scaling, shearing, reflection, rotation, and cropping by defining corresponding functions:

1.Translation moves the image along the x or y-axis. 2.Scaling resizes the image by scaling factors. 3.Shearing distorts the image along one axis. 4.Reflection flips the image horizontally or vertically. 5.Rotation rotates the image by a given angle.

### Step4:
Display the transformed images using Matplotlib for visualization. Convert the BGR image to RGB format to ensure proper color representation.

### Step5:
Save or display the final transformed images for analysis and use plt.show() to display them inline in Jupyter or compatible environments.

## Program:
```python
Developed By:P.Senthil Arunachalam
Register Number:212224240147
```

# import the libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image and check if it's loaded correctly
input_img = cv2.imread("dog.jpg")
if input_img is None:
    print("Error: Image not found.")
else:
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

#Display the image
plt.figure(figsize=(8, 8))  # Optional: adjust size for better clarity
plt.axis('off')
plt.imshow(input_img)
plt.title("Original Image")
plt.tight_layout()  # Optional: makes sure layout is tight
plt.show()

<Figure size 800x800 with 1 Axes><img width="790" height="553" alt="image" src="https://github.com/user-attachments/assets/080ee9ef-1278-444b-86ea-b2007bbd97de" />
 
# Get image dimensions
 
rows, cols, dim = input_img.shape

# Define translation matrix

translation_x = 50  # Move 50 pixels to the right
translation_y = 50  # Move 50 pixels down
M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

# Perform the translation

translated_img = cv2.warpAffine(input_img, M, (cols, rows))

# Display the translated image

plt.axis('off')
plt.imshow(translated_img)
plt.title("Translated Image")
plt.show()

<Figure size 640x480 with 1 Axes><img width="515" height="370" alt="image" src="https://github.com/user-attachments/assets/687f28a2-f8c8-4860-838a-f752f3e63d05" />

# Perform scaling with different scaling factors for x and y axes

fx, fy = 1.5, 2.0  # Scaling factors
scaled_img = cv2.resize(input_img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

# Show the scaled image

plt.axis('off')
plt.imshow(scaled_img)
plt.title("Scaled Image")
plt.show()

<Figure size 640x480 with 1 Axes><img width="435" height="409" alt="image" src="https://github.com/user-attachments/assets/c039fd4a-a787-447b-9f3b-db5e91b67511" />
    
# Shearing in X-axis

shear_factor_x = 0.5  # Shear factor for X-axis
M_x = np.float32([[1, shear_factor_x, 0], [0, 1, 0]])

# Apply shearing
sheared_img_xaxis = cv2.warpAffine(input_img, M_x, (int(cols + shear_factor_x * rows), rows))

# Display the sheared image

plt.axis('off')
plt.title("Sheared X-axis")
plt.imshow(sheared_img_xaxis)

<Figure size 640x480 with 1 Axes><img width="515" height="288" alt="image" src="https://github.com/user-attachments/assets/4e22b34c-7500-4214-8a89-890d4b30dab1" />

# Flip image along the X-axis (vertical reflection)

reflected_img_xaxis = cv2.flip(input_img, 0)

# Display the reflected image

plt.axis("off")
plt.title("Reflected (X-axis)")
plt.imshow(reflected_img_xaxis)

<Figure size 640x480 with 1 Axes><img width="515" height="370" alt="image" src="https://github.com/user-attachments/assets/8dec603a-ce61-4032-bec5-7079b84d3ec5" />

# Get image dimensions

rows, cols = input_img.shape[:2]

# Define rotation center, angle, and scale

center = (cols // 2, rows // 2)
angle = 45  # Rotation angle in degrees
scale = 1.0  # Scale factor (1.0 means no scaling)

# Get the rotation matrix

M = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation

rotated_img = cv2.warpAffine(input_img, M, (cols, rows))

# Display the rotated image

plt.axis('off')
plt.title("Rotated Image (45°)")
plt.imshow(rotated_img)
plt.show()

<Figure size 640x480 with 1 Axes><img width="515" height="370" alt="image" src="https://github.com/user-attachments/assets/fb334577-b0ed-4408-8bd2-42d7aa34973b" />

# Define the cropping coordinates: top-left corner (x, y) and width (w), height (h)
x, y, w, h = 400, 100, 1000, 500

# Crop the image by slicing the array
cropped_image = input_img[y:y+h, x:x+w]

# Display the cropped image (convert to RGB for correct color representation)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Cropped Image")
plt.axis('off')
plt.show()

<Figure size 640x480 with 1 Axes><img width="515" height="288" alt="image" src="https://github.com/user-attachments/assets/890f6ee7-9cba-4979-8a94-e0bb1529fc98" />

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
