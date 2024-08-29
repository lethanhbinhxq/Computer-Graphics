from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import math

import pandas as pd

# Coordinate of 4 vertices in a tetrahedron
vertices = np.array([
[0.0, 0.0, 1.0], [0.0, 0.942809, -0.33333],
[-0.816497, -0.471405, -0.333333],
[0.816497, -0.471405, -0.333333]
])

center = np.mean(vertices, axis=0)
def calculate_normal(vertex):
    normal = vertex - center
    return normal / np.linalg.norm(normal)

def calculate_color(vertex, R):
    return (vertex + R) / (2 * R)

def calculate_texture(vertex, R):
    normal = calculate_normal(vertex)
    normal = normal.ravel()
    x, y, z = normal[0], normal[1], normal[2]
    u = math.atan2(z, x) / (2 * math.pi)
    if u < 0:
        u += 1
    v = math.asin(y) / math.pi + 0.5
    return [u, v]

def handle_seam(v1, v2, v3, R):
    # Calculate texture coordinates
    t1 = calculate_texture(v1, R)
    t2 = calculate_texture(v2, R)
    t3 = calculate_texture(v3, R)

    # Check if there's a seam crossing
    # if abs(t1[0] - t2[0]) > 0.5 or abs(t2[0] - t3[0]) > 0.5 or abs(t3[0] - t1[0]) > 0.5:
    #     # Adjust texture coordinates
    #     if t1[0] < 0.5:
    #         t1[0] += 1
    #     if t2[0] < 0.5:
    #         t2[0] += 1
    #     if t3[0] < 0.5:
    #         t3[0] += 1
    return t1, t2, t3

def triangle(A, B, C, R, ntriangles: list, data: list):
    tA, tB, tC = handle_seam(A, B, C, R)
    A_attr = ((R*A).ravel().tolist() + calculate_normal(A).ravel().tolist() + calculate_color(A, R).ravel().tolist()
              + tA)
    B_attr = ((R*B).ravel().tolist() + calculate_normal(B).ravel().tolist() + calculate_color(B, R).ravel().tolist()
              + tB)
    C_attr = ((R*C).ravel().tolist() + calculate_normal(C).ravel().tolist() + calculate_color(C, R).ravel().tolist()
              + tC)
    keys = ['v.x', 'v.y', 'v.z', 'n.x', 'n.y', 'n.z', 'c.x', 'c.y', 'c.z', 't.x', 't.y']
    data.append(dict(zip(keys, A_attr)))
    data.append(dict(zip(keys, B_attr)))
    data.append(dict(zip(keys, C_attr)))
    ntriangles[0] += 1

def face(A, B, C, n, R, ntriangles: list, data: list):
    # A = [a1, a2, a3]
    # B = [b1, b2, b3]
    # C = [c1, c2, c3]
    A = A.reshape((1,-1)); B = B.reshape((1, -1)); C = C.reshape((1, -1))
    # Do the division recursively, each time decrease n by 1 until n = 0
    if n > 0:
        #          A
        #
        #
        #   V[0]     V[1]
        #
        #
        # B      V[2]      C
        V = np.concatenate((A + B, A + C, B + C), axis = 0)
        V = V/np.linalg.norm(V, axis=1, keepdims=True)
        # Continue to do the division with 4 small triangles we have divided above
        face(A, V[0], V[1], n - 1, R, ntriangles, data)
        face(C, V[1], V[2], n - 1, R,  ntriangles, data)
        face(B, V[2], V[0], n - 1, R,  ntriangles, data)
        face(V[0], V[2], V[1], n - 1, R,  ntriangles, data)
    # In base case (n = 0), store the value for 3 vertices of the triangle in the csv file
    else:
        triangle(A, B, C, R, ntriangles, data)

def sphere(n, R, ntriangles: list, data: list):
    # Each is the number of divided triangles in each face of the initial tetrahedron
    ntriangles1 = [0]; ntriangles2 = [0]; ntriangles3 = [0]; ntriangles4 = [0]
    # Recursively divide each face of the tetrahedron into smaller triangles
    face(vertices[0], vertices[1], vertices[2], n, R, ntriangles1, data)
    face(vertices[1], vertices[2], vertices[3], n, R, ntriangles2, data)
    face(vertices[0], vertices[2], vertices[3], n, R, ntriangles3, data)
    face(vertices[0], vertices[3], vertices[1], n, R, ntriangles4, data)
    # Store the number of divided triangles in each face of the tetrahedron into "ntriangles" and return the result
    ntriangles[0] = ntriangles1[0]
    ntriangles[1] = ntriangles2[0]
    ntriangles[2] = ntriangles3[0]
    ntriangles[3] = ntriangles4[0]

def find_center(data):
    # Extract the vertex positions
    vertices = np.array([[d['v.x'], d['v.y'], d['v.z']] for d in data])
    # Compute the mean of the vertices
    center = np.mean(vertices, axis=0)
    return center

def main():
    ntriangles = [0, 0, 0, 0]
    data = []
    sphere(5, 1, ntriangles, data)
    print(ntriangles)
    df = pd.DataFrame(data)
    print('output:')
    df.to_csv('./sphere2.csv', index=False)

main()
