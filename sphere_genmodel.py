from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

import pandas as pd

vertices = np.array([
[0.0, 0.0, 1.0], [0.0, 0.942809, -0.33333],
[-0.816497, -0.471405, -0.333333],
[0.816497, -0.471405, -0.333333]
])

def triangle(A, B, C, R, ntriangles: list, data: list):
    A_attr = R*A.ravel().tolist() + A.ravel().tolist()
    B_attr = R*B.ravel().tolist() + B.ravel().tolist()
    C_attr = R*C.ravel().tolist() + C.ravel().tolist()
    keys = ['v.x', 'v.y', 'v.z', 'n.x', 'n.y', 'n.z']
    data.append(dict(zip(keys, A_attr)))
    data.append(dict(zip(keys, B_attr)))
    data.append(dict(zip(keys, C_attr)))
    ntriangles[0] += 1

def face(A, B, C, n, R, ntriangles: list, data: list):
    A = A.reshape((1,-1)); B = B.reshape((1, -1)); C = C.reshape((1, -1))
    if n > 0:
        V = np.concatenate((A + B, A + C, B + C), axis = 0)
        V = V/np.linalg.norm(V, axis=1, keepdims=True)
        face(A, V[0], V[1], n - 1, R, ntriangles, data)
        face(C, V[1], V[2], n - 1, R,  ntriangles, data)
        face(B, V[2], V[0], n - 1, R,  ntriangles, data)
        face(V[0], V[2], V[1], n - 1, R,  ntriangles, data)
    else:
        triangle(A, B, C, R, ntriangles, data)

def sphere(n, R, ntriangles: list, data: list):
    ntriangles1 = [0]; ntriangles2 = [0]; ntriangles3 = [0]; ntriangles4 = [0]

    face(vertices[0], vertices[1], vertices[2], n, R, ntriangles1, data)
    face(vertices[3], vertices[2], vertices[1], n, R, ntriangles2, data)
    face(vertices[0], vertices[3], vertices[1], n, R, ntriangles3, data)
    face(vertices[0], vertices[2], vertices[3], n, R, ntriangles4, data)
    ntriangles[0] = ntriangles1[0]
    ntriangles[1] = ntriangles2[0]
    ntriangles[2] = ntriangles3[0]
    ntriangles[3] = ntriangles4[0]

def main():
    ntriangles = [0, 0, 0, 0]
    data = []
    sphere(5, 3, ntriangles, data)
    print(ntriangles)

    # line0 = [ntriangles[0]] + [None]*5
    # line1 = [ntriangles[1]] + [None] * 5
    # line2 = [ntriangles[2]] + [None] * 5
    # line3 = [ntriangles[3]] + [None] * 5
    # keys = ['v.x', 'v.y', 'v.z', 'n.x', 'n.y', 'n.z']
    # data = [dict(zip(keys, line0)),
    #         dict(zip(keys, line1)),
    #         dict(zip(keys, line2)),
    #         dict(zip(keys, line3)),
    #         ] + data

    df = pd.DataFrame(data)
    print('output:')
    df.to_csv('./sphere.csv', index=False)

main()