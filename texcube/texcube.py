from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import glfw
import math


"""
TOP (y =+1): EFGH
                                |
   G (-1, +1, -1)   ......................... H: (+1, +1, -1) 
   color: (0,1,1)   |           |           |    WHITE: (1, 1, 1)
                    |           |           |
                    |           |           |
            --------------------------------------->X
                    |           |           |
                    |           |           |
                    |           |           |
   F: (-1, +1, +1)  ......................... E: (+1, +1, +1)
   BLUE: (0, 0, 1)              |              color: (1,0,1)
                                V 
                                Z

BOTTOM (y=-1): ABCD
                                |
    C: (-1, -1, -1) ......................... D: (+1, -1, -1)
    GREEN: (0,1,0)  |           |           |  color: (1,1,0)
                    |           |           |
                    |           |           |
            --------------------------------------->X
                    |           |           |
                    |           |           |
                    |           |           |
    B: (-1, -1, +1) ......................... A: (+1, -1, +1)
    BLACK: (0,0,0)              |               RED: (1,0,0)
                                V 
                                Z

Texture (2D image: 3x4, see: shape/texcube/image/texture.jpeg
        0             1/4             2/4             3/4             1.0  
   0    ...............................F...............E.......................>X
        |              |               |               |               |
        |              |               |               |               |
        |              |               |               |               |
   1/3  E..............F...............G...............H...............E
        |              |               |               |               |
        |              |               |               |               |
        |              |               |               |               |
   2/3  A..............B...............C...............D...............A
        |              |               |               |               |
        |              |               |               |               |
        |              |               |               |               |
   1.0  ...............................B...............A................
        |
        V 
        Y
"""


class TexCube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array(
            [
                #  for SIDE faces
                [+1, -1, +1],  # A  0
                [+1, +1, +1],  # E  1
                [-1, -1, +1],  # B  2
                [-1, +1, +1],  # F  3
                [-1, -1, -1],  # C  4
                [-1, +1, -1],  # G  5
                [+1, -1, -1],  # D  6
                [+1, +1, -1],  # H  7
                [+1, -1, +1],  # A  8: repeated (for texturing),
                [+1, +1, +1],  # E  9: repeated (for texturing),
                # for TOP face
                [-1, +1, -1],  # G  10
                [-1, +1, +1],  # F  11
                [+1, +1, -1],  # H  12
                [+1, +1, +1],  # E  13
                # for BOTTOM face
                [-1, -1, +1],  # B  14
                [-1, -1, -1],  # C  15
                [+1, -1, +1],  # A  16
                [+1, -1, -1]  # D  17
            ],
            dtype=np.float32
        )

        # concatenate three sequences of triangle strip: [0 - 9] [10 - 13] [14-17]
        # => repeat 9, 10, 13, 14
        self.indices = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 14, 14, 15, 16, 17],
            dtype=np.int32
        )

        self.normals = self.vertices.copy()
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)

        # texture coordinates
        self.texcoords = np.array(
            [
                # for SIDE faces
                0, 2 / 3,  # A
                0, 1 / 3,  # E
                   1 / 4, 2 / 3,  # B
                   1 / 4, 1 / 3,  # F
                   2 / 4, 2 / 3,  # C
                   2 / 4, 1 / 3,  # G
                   3 / 4, 2 / 3,  # D
                   3 / 4, 1 / 3,  # H
                1.0, 2 / 3,  # A: repeated (same vertices, but different texture coordinates),
                1.0, 1 / 3,  # E: repeated (same vertices, but different texture coordinates),
                # for TOP face
                   2 / 4, 1 / 3,  # G
                   2 / 4, 0,  # F
                   3 / 4, 1 / 3,  # H
                   3 / 4, 0,  # E
                # for BOTTOM face
                   2 / 4, 1.0,  # B
                   2 / 4, 2 / 3,  # C
                   3 / 4, 1.0,  # A
                   3 / 4, 2 / 3  # D
            ],
            dtype=np.float32
        )

        self.colors = np.array(
            [
                #  for SIDE faces
                [1, 0, 0],  # A  0
                [1, 0, 1],  # E  1
                [0, 0, 0],  # B  2
                [0, 0, 1],  # F  3
                [0, 1, 0],  # C  4
                [0, 1, 1],  # G  5
                [1, 1, 0],  # D  6
                [1, 1, 1],  # H  7
                [1, 0, 0],  # A  8: repeated (for texturing),
                [1, 0, 1],  # E  9: repeated (for texturing),
                # for TOP face
                [0, 1, 1],  # G  10
                [0, 0, 1],  # F  11
                [1, 1, 1],  # H  12
                [1, 0, 1],  # E  13
                # for BOTTOM face
                [0, 0, 0],  # B  14
                [0, 1, 0],  # C  15
                [1, 0, 0],  # A  16
                [1, 1, 0]  # D  17
            ],
            dtype=np.float32
        )

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao.add_vbo(3, self.normals, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        # setup textures
        self.uma.setup_texture("texture", "./image/texture.jpeg")

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        # Materials
        K_materials = np.array([
            [0.5, 0.0, 0.7],  # diffuse
            [0.5, 0.0, 0.7],  # specular
            [0.5, 0.0, 0.7]  # ambient
        ], dtype=np.float32)

        shininess = 100.0
        phong_factor = 0.3  # blending factor for phong shading and texture

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
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

class Sphere1(object):
    def __init__(self, vert_shader, frag_shader):
        N = 300
        r = 0.5
        self.vertices = []
        self.colors = []
        self.texcoord = []
        for i in range(N):
            theta1 = (math.pi * i) / N
            theta2 = (math.pi * (i + 1)) / N
            for j in range(N + 1):
                phi = (2 * math.pi * j) / N
                x = r * math.cos(phi) * math.sin(theta1)
                y = r * math.sin(phi) * math.sin(theta1)
                z = r * math.cos(theta1)
                self.vertices.extend([x, y, z])

                color_r = (abs(x) + r) / (2 * r)
                color_g = (abs(y) + r) / (2 * r)
                color_b = (abs(z) + r) / (2 * r)
                self.colors.extend([color_r, color_g, color_b])

                tx = j / N
                ty = i / N
                self.texcoord.extend([tx, ty])

                x = r * math.cos(phi) * math.sin(theta2)
                y = r * math.sin(phi) * math.sin(theta2)
                z = r * math.cos(theta1)
                self.vertices.extend([x, y, z])

                color_r = (abs(x) + r) / (2 * r)
                color_g = (abs(y) + r) / (2 * r)
                color_b = (abs(z) + r) / (2 * r)
                self.colors.extend([color_r, color_g, color_b])

                tx = j / N
                ty = (i + 1) / N
                self.texcoord.extend([tx, ty])
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)
        self.texcoord = np.array(self.texcoord, dtype=np.float32)

        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        # create index with np.unit32
        self.indices = np.arange(len(self.vertices)).astype(np.uint32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.texcoord, ncomponents=2, stride=0, offset=None)
        self.vao.add_vbo(3, self.normals, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        # setup textures
        self.uma.setup_texture("texture", "./image/earth.jpg")

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        # Materials
        K_materials = np.array([
            [0.5, 0.0, 0.7],  # diffuse
            [0.5, 0.0, 0.7],  # specular
            [0.5, 0.0, 0.7]  # ambient
        ], dtype=np.float32)

        shininess = 100.0
        phong_factor = 0.3  # blending factor for phong shading and texture

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
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