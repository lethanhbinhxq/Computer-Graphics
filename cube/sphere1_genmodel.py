from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import math

import pandas as pd

# phi: vertical -> 2pi
# theta: horizontal -> pi
def sphere(n, R, data: list):
    for i in range(n):
        theta1 = (math.pi * i) / n
        theta2 = (math.pi * (i + 1)) / n

        def point_data(theta, i, j):
            phi = (2 * math.pi * j) / n
            vx = R * math.cos(phi) * math.sin(theta)
            vy = R * math.sin(phi) * math.sin(theta)
            vz = R * math.cos(theta)
            nx = vx / R
            ny = vy / R
            nz = vz / R
            cx = (vx + R) / (2 * R)
            cy = (vy + R) / (2 * R)
            cz = (vz + R) / (2 * R)
            tx = j / (n + 1)
            ty = i / n
            return [vx, vy, vz, nx, ny, nz, cx, cy, cz, tx, ty]

        for j in range(n + 1):
            point1 = point_data(theta1, i, j)
            point2 = point_data(theta2, i + 1, j)
            keys = ['v.x', 'v.y', 'v.z', 'n.x', 'n.y', 'n.z', 'c.x', 'c.y', 'c.z', 't.x', 't.y']
            values = np.array(point1)
            data.append(dict(zip(keys, values)))
            values = np.array(point2)
            data.append(dict(zip(keys, values)))

def main():
    data = []
    sphere(100, 1, data)
    df = pd.DataFrame(data)
    df.to_csv('./sphere1.csv', index=False)

main()