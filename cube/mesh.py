import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import glfw
import math

class Mesh3D(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = []
        self.colors = []
        self.texture = []
        def f(x, z, colorFactor):
            # calculate vertex
            y = x ** 4 - 2 * x ** 2 + z ** 4
            self.vertices.append([x, y, z])
            # calculate color
            r = (x + colorFactor) / (2 * colorFactor)
            g = (y + colorFactor) / (2 * colorFactor)
            b = (z + colorFactor) / (2 * colorFactor)
            self.colors.extend([r, g, b])
            # calculate texture
            u = (x + 2) / 4
            v = (y + 2) / 4
            self.texture.append([u, v])
        for x in np.linspace(-2, 2, 100):
            for z in np.linspace(-2, 2, 100):
                f(x, z, 1)
                f(x + 0.1, z, 1)

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)
        self.texture = np.array(self.texture, dtype=np.float32)

        self.indices = np.arange(len(self.vertices)).astype(np.uint32)

        # save indices of vertices for each triangle
        self.triangles = []
        for i in range(2, len(self.indices), 1):
            if i % 2 == 0:
                self.triangles.append((i - 2, i - 1, i))
            else:
                self.triangles.append((i - 2, i, i - 1))

        self.normals = [[] for _ in range(len(self.vertices))]
        for triangle in self.triangles:
            idx0, idx1, idx2 = triangle # index of each vertex
            v0, v1, v2 = self.vertices[idx0], self.vertices[idx1], self.vertices[idx2] # vertex
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)
            self.normals[idx0].append(normal)
            self.normals[idx1].append(normal)
            self.normals[idx2].append(normal)
        self.normals = [np.average(normal, axis=0) for normal in self.normals]
        self.normals = np.array(self.normals, dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # # setup VAO for drawing cylinder's side
        # self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        # self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)

        # setup VAO for drawing (texture mapping)
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.texture, ncomponents=2, stride=0, offset=None)
        # self.vao.add_vbo(3, self.colors, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        # setup textures
        self.uma.setup_texture("texture", "./image/leaf.jpg")

        normalMat = np.identity(4, 'f')
        projection = T.ortho(-0.5, 2.5, -0.5, 1.5, -1, 1)
        modelview = np.identity(4, 'f')

        # Light
        I_light = np.array([
            [0.5, 0.5, 0.5],  # diffuse
            [0.5, 0.5, 0.5],  # specular
            [0.5, 0.5, 0.5]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        # Materials
        K_materials = np.array([
            [0.6, 1, 1],  # diffuse
            [0.6, 1, 1],  # specular
            [0.6, 1, 1]  # ambient
        ], dtype=np.float32)

        shininess = 100.0
        # mode = 1
        phong_factor = 0.3

        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        # self.uma.upload_uniform_scalar1i(mode, 'mode')
        self.uma.upload_uniform_scalar1f(phong_factor, 'phong_factor')

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2