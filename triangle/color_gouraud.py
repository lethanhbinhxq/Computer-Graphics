import numpy as np
import matplotlib.pyplot as plt

# Homework: color interpolation by gourad and phong model
def area(A, B, C):
    AB = np.subtract(B, A)
    AC = np.subtract(C, A)
    cross_product = np.cross(AB, AC)
    area = 0.5 * np.linalg.norm(cross_product)
    return area

def calculate_color(A, B, C, colorA, colorB, colorC, P):
    S_PAB = area(P, A, B)
    S_PBC = area(P, B, C)
    S_PCA = area(P, C, A)
    S_ABC = area(A, B, C)
    a = S_PBC / S_ABC
    b = S_PCA / S_ABC
    c = S_PAB / S_ABC
    colorP = a * np.array(colorA) + b * np.array(colorB) + c * np.array(colorC)
    return colorP

def is_inside(P, A, B, C):
    S_PAB = area(P, A, B)
    S_PBC = area(P, B, C)
    S_PCA = area(P, C, A)
    S_ABC = area(A, B, C)
    return np.isclose(S_ABC, S_PAB + S_PBC + S_PCA)

# Define the size of the image
width = 200
height = 200

# Create an empty image with white background
image = np.ones((height, width, 3), dtype=np.float32)

# Define the coordinates of the triangle vertices
vertices = np.array([[0.5 * (width - 1), 0], [0, (height - 1)], [(width - 1), (height - 1)]])

# Define the colors for the vertices in RGB format
colors = np.array([[1, 0, 0],  # Red
                   [0, 1, 0],  # Green
                   [0, 0, 1]]) # Blue

A = vertices[0]
B = vertices[1]
C = vertices[2]

colorA = colors[0]
colorB = colors[1]
colorC = colors[2]

X = np.linspace(0, width - 1, 200)
Y = np.linspace(0, height - 1, 200)
x, y = np.meshgrid(X, Y)
points = np.column_stack((x.flatten(), y.flatten()))
for point in points:
  if is_inside(point, A, B, C):
    vertices = np.vstack([vertices, point])
    colorP = calculate_color(A, B, C, colorA, colorB, colorC, point)
    colors = np.vstack([colors, colorP])

# Convert the coordinates to integers
vertices = np.round(vertices).astype(int)

# Fill the triangle with colors by modifying pixel values directly
for i in range(len(vertices)):
    image[vertices[i, 1], vertices[i, 0]] = colors[i]

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()


"""
texture: image
tP = a*tA + b*tB + c*tC
tP = tP.clip(0, 1)
tP = (tP*texture.shape[:2]).astype(np.int64)
tP = (tP - 1).clip(0, tP.max())
cPT = texture[tP[:,0],tP[:,1],:]
cPT[mask] = [100,100,100] 
"""