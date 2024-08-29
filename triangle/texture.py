import numpy as np
import cv2
import matplotlib.pyplot as plt

def area(A, B, C):
    AB = np.subtract(B, A)
    AC = np.subtract(C, A)
    cross_product = np.cross(AB, AC)
    area = 0.5 * np.linalg.norm(cross_product)
    return area

def is_inside(P, A, B, C):
    S_PAB = area(P, A, B)
    S_PBC = area(P, B, C)
    S_PCA = area(P, C, A)
    S_ABC = area(A, B, C)
    return np.isclose(S_ABC, S_PAB + S_PBC + S_PCA)

# Load the texture image
texture_image = cv2.imread('5295209.jpg')
texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

# Define vertices of the triangle
A = np.array([100, 100])
B = np.array([300, 100])
C = np.array([200, 300])

x, y = np.meshgrid(np.linspace(100, 300, 256), np.linspace(100, 300, 256))

# Flatten the grid to get a list of points
points = np.column_stack((x.flatten(), y.flatten()))

# Create an empty image to draw the triangle with texture
triangle_texture = np.zeros_like(texture_image)

# Iterate over each pixel in the triangle and map the texture
for i, point in enumerate(points):
    if is_inside(point, A, B, C):
        # Calculate barycentric coordinates
        v0 = C - A
        v1 = B - A
        v2 = point - A

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Map the texture
        tex_x = int(u * (texture_image.shape[1] - 1))
        tex_y = int(v * (texture_image.shape[0] - 1))
        triangle_texture[i // 256, i % 256] = texture_image[tex_y, tex_x]

# Plot the triangle with texture
plt.imshow(triangle_texture)
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.axis('off')
plt.show()