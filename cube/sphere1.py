import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import glfw
import math

class Sphere1(object):
    def __init__(self, vert_shader, frag_shader):
        df = pd.read_csv('sphere1.csv')

        self.vertices = df[['v.x', 'v.y', 'v.z']].values.tolist()
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.normals = df[['n.x', 'n.y', 'n.z']].values.tolist()
        self.normals = np.array(self.normals, dtype=np.float32)

        self.colors = df[['c.x', 'c.y', 'c.z']].values.tolist()
        self.colors = np.array(self.colors, dtype=np.float32)

        self.texture = df[['t.x', 't.y']].values.tolist()
        self.texture = np.array(self.texture, dtype=np.float32)
        self.texture.flatten()

        # create index with np.unit32
        self.indices = np.arange(len(self.vertices)).astype(np.uint32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        # setup VAO for drawing (texture mapping)
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.texture, ncomponents=2, stride=0, offset=None)
        # self.vao.add_vbo(3, self.colors, ncomponents=3, stride=0, offset=None)

        # # setup VAO for drawing (phong shading)
        # self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        # self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)


        # setup EBO for drawing
        self.vao.add_ebo(self.indices)

        # setup textures
        # self.uma.setup_texture("texture", "./image/earth.jpg")
        self.uma.setup_texture("texture", "./image/earth.jpg")

        normalMat = np.identity(4, 'f')
        projection = T.ortho(-0.5, 2.5, -0.5, 1.5, -1, 1)
        modelview = np.identity(4, 'f')

        # Light
        I_light = np.array([
            [1, 1, 1],  # diffuse
            [1, 1, 1],  # specular
            [1, 1, 1]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0, 0, 0.5]  # ambient
        ], dtype=np.float32)

        shininess = 100.0
        # mode = 1
        phong_factor = 0.2

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
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(self.indices), GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2