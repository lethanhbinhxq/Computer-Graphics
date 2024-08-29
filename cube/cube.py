import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import glfw
import math


class Cube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array(
            [
                [-1, -1, +1],  # A <= Bottom: ABCD
                [+1, -1, +1],  # B
                [+1, -1, -1],  # C
                [-1, -1, -1],  # D
                [-1, +1, +1],  # E <= Top: EFGH
                [+1, +1, +1],  # F
                [+1, +1, -1],  # G
                [-1, +1, -1],  # H
            ],
            dtype=np.float32
        )

        self.indices = np.array(
            [0, 4, 1, 5, 2, 6, 3, 7, 0, 4, 4, 0, 0, 3, 1, 2, 2, 4, 4, 7, 5, 6],
            dtype=np.int32
        )

        self.normals = self.vertices.copy()
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)

        # colors: RGB format
        self.colors = np.array(
            [  # R    G    B
                [1.0, 0.0, 0.0],  # A <= Bottom: ABCD
                [1.0, 0.0, 1.0],  # B
                [0.0, 0.0, 1.0],  # C
                [0.0, 0.0, 0.0],  # D
                [1.0, 1.0, 0.0],  # E <= Top: EFGH
                [1.0, 1.0, 1.0],  # F
                [0.0, 1.0, 1.0],  # G
                [0.0, 1.0, 0.0],  # H
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
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

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

# y = x^4 - 2x^2 + z^2
# x, z: [-2, 2]

class Mesh3D(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = []
        self.colors = []
        for x in np.linspace(-2, 2, 100):
            for z in np.linspace(-2, 2, 100):
                # y = x**4 - 2 * x**2 + z**2
                y = x ** 2 + z ** 2
                self.vertices.append([x, y, z])
                color_r = (x + 2) / 4
                color_g = (y + 2) / 4
                color_b = (z + 2) / 4
                self.colors.extend([color_r, color_g, color_b])

                # norm = (y + 25) / 50
                # self.colors.append([1.0 - norm, 0.0, norm])
                #
                y = (x + 1) ** 2 + z**2
                self.vertices.append([x + 1, y, z])
                #
                # self.colors.append([1.0 - norm, 0.0, norm])

                color_r = (x + 1 + 2) / 4
                color_g = (y + 2) / 4
                color_b = (z + 2) / 4
                self.colors.extend([color_r, color_g, color_b])

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)

        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.indices = np.arange(len(self.vertices)).astype(np.uint32)

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
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        normalMat = np.identity(4, 'f')
        projection = T.ortho(-0.5, 2.5, -0.5, 1.5, -1, 1)
        modelview = np.identity(4, 'f')

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        shininess = 100.0
        mode = 1

        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')

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

class Cone(object):
    def __init__(self, vert_shader, frag_shader):

        N = 50
        alpha = np.linspace(0, 2 * np.pi, N).astype(np.float32)
        x = np.cos(alpha).astype(np.float32).reshape(-1, 1)
        y = np.sin(alpha).astype(np.float32).reshape(-1, 1)
        z = np.zeros_like(y).astype(np.float32)
        o = np.array([0, 0, 0], dtype=np.float32).reshape(1, -1)
        self.bottom_vertices = np.hstack([x, y, z])
        self.bottom_vertices = np.vstack([o, self.bottom_vertices])

        a = np.array([0, 0, 1], dtype=np.float32).reshape(1, -1)
        self.side_vertices = np.hstack([x, y, z])
        self.side_vertices = np.vstack([a, self.side_vertices])

        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        # create index with np.unit32
        self.bottom_indices = np.arange(len(self.bottom_vertices)).astype(np.uint32)
        self.side_indices = np.arange(len(self.side_vertices)).astype(np.uint32)

        # colors: RGB format
        bottom_colors = np.array([
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        side_color = np.array([
            [1.0, 0.0, 1.0]
        ], dtype=np.float32)
        side_colors = np.tile(side_color, (len(self.side_vertices), len(self.side_vertices))).reshape(-1, 3)
        self.colors = np.vstack([bottom_colors, side_colors])
        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.bottom_vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(0, self.side_vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.bottom_indices)
        self.vao.add_ebo(self.side_indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, len(self.bottom_indices), GL.GL_UNSIGNED_INT, None)
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, len(self.side_indices), GL.GL_UNSIGNED_INT, None)


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
        for i in range(N):
            theta1 = (math.pi * i) / N
            theta2 = (math.pi * (i + 1)) / N
            for j in range(N + 1):
                phi = (2 * math.pi * j) / N
                x = r * math.cos(phi) * math.sin(theta1)
                y = r * math.cos(theta1)
                z = r * math.sin(phi) * math.sin(theta1)
                self.vertices.extend([x, y, z])

                color_r = (abs(x) + r) / (2 * r)
                color_g = (abs(y) + r) / (2 * r)
                color_b = (abs(z) + r) / (2 * r)
                self.colors.extend([color_r, color_g, color_b])

                x = r * math.cos(phi) * math.sin(theta2)
                y = r * math.cos(theta2)
                z = r * math.sin(phi) * math.sin(theta2)
                self.vertices.extend([x, y, z])

                color_r = (abs(x) + r) / (2 * r)
                color_g = (abs(y) + r) / (2 * r)
                color_b = (abs(z) + r) / (2 * r)
                self.colors.extend([color_r, color_g, color_b])
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)

        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        # create index with np.unit32
        self.indices = np.arange(len(self.vertices)).astype(np.uint32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

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

class Sphere2(object):
    def __init__(self, vert_shader, frag_shader):
        df = pd.read_csv('sphere2.csv')
        self.vertices = df[['v.x', 'v.y', 'v.z']].values.tolist()
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.normals = df[['n.x', 'n.y', 'n.z']].values.tolist()
        self.normals = np.array(self.normals, dtype=np.float32)

        # color_r = (abs(x) + r) / (2 * r)
        # color_g = (abs(y) + r) / (2 * r)
        # color_b = (abs(z) + r) / (2 * r)
        # self.colors.extend([color_r, color_g, color_b])

        # self.colors = []
        # for v in self.vertices:
        #     r = (v[0] + 2) / 4
        #     g = (v[1] + 2) / 4
        #     b = (v[2] + 2) / 4
        #     self.colors.extend(([r, g, b]))
        # self.colors = np.array(self.colors, dtype=np.float32)

        self.colors = df[['c.x', 'c.y', 'c.z']].values.tolist()
        self.colors = np.array(self.colors, dtype=np.float32)

        # create index with np.unit32
        self.indices = np.arange(len(self.vertices)).astype(np.uint32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

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

class Test(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [0.0, 0.0, 1.0], [0.0, 0.942809, -0.33333],
            [-0.816497, -0.471405, -0.333333],
            [0.816497, -0.471405, -0.333333]
        ], dtype=np.float32)
        self.colors = np.array([
            [0.0, 0.0, 1.0], [0.0, 0.942809, -0.33333],
            [-0.816497, -0.471405, -0.333333],
            [0.816497, -0.471405, -0.333333]
        ], dtype=np.float32)

        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.indices = np.arange(len(self.vertices)).astype(np.uint32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(self.indices), GL.GL_UNSIGNED_INT, None)

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

class Sphere3(object):
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