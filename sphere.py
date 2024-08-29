from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pandas as pd


# Load vertex and normal data from CSV file
def load_sphere_data(filename):
    df = pd.read_csv(filename)
    vertices = df[['v.x', 'v.y', 'v.z']].values.tolist()
    normals = df[['n.x', 'n.y', 'n.z']].values.tolist()
    return vertices, normals


def draw_sphere(vertices, normals):
    glBegin(GL_POINTS)
    for i in range(0, len(vertices), 3):
        for j in range(3):
            glNormal3fv(normals[i + j])
            glVertex3fv(vertices[i + j])
    glEnd()


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, 800 / 600, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glColor3f(1.0, 1.0, 1.0)
    draw_sphere(vertices, normals)

    glutSwapBuffers()


def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)


if __name__ == "__main__":
    filename = 'sphere.csv'  # Path to the sphere CSV file
    vertices, normals = load_sphere_data(filename)

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Loaded Sphere")
    init()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)

    glutMainLoop()
