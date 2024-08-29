from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import glfw
import cynlinder.creation as generator

class TexturedCylinder(object):
    def __init__(self, vert_shader, frag_shader):
        """
        self.side_data, self.bottom_data, self.top_data:
        each row: v.x, v.y, v.z, t.x, t.y, n.x, n.y, n.z
        """
        self.side_data, self.side_indices, \
        self.bottom_data, self.bottom_indices, \
        self.top_data, self.top_indices = generator.generate(nsegments=50, height=1)

        self.side_vao = VAO()
        self.bottom_vao = VAO()
        self.top_vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #
        self.selected_texture = 1

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # 8: 8 coordinates between two consecutive vertices/normals/texture-coords
        # 4: each coordinate costs 4 bytes
        stride = 8*4
        offset_v = ctypes.c_void_p(0)  # vertices stored first
        offset_t = ctypes.c_void_p(3*4)  # store 3 coordinates of a vertex, then store its texture coord
        offset_n = ctypes.c_void_p(5*4)  # 5: 3 coordinates of a vertex + 2 texture coords

        # setup VAO for drawing cylinder's side
        self.side_vao.add_vbo(0, self.side_data, ncomponents=3, stride=stride, offset=offset_v)
        self.side_vao.add_vbo(1, self.side_data, ncomponents=3, stride=stride, offset=offset_n)
        self.side_vao.add_vbo(2, self.side_data, ncomponents=2, stride=stride, offset=offset_t)

        # setup VAO for drawing cylinder's bottom
        self.bottom_vao.add_vbo(0, self.bottom_data, ncomponents=3, stride=stride, offset=offset_v)
        self.bottom_vao.add_vbo(1, self.bottom_data, ncomponents=3, stride=stride, offset=offset_n)
        self.bottom_vao.add_vbo(2, self.bottom_data, ncomponents=2, stride=stride, offset=offset_t)

        # setup VAO for drawing cylinder's top
        self.top_vao.add_vbo(0, self.top_data, ncomponents=3, stride=stride, offset=offset_v)
        self.top_vao.add_vbo(1, self.top_data, ncomponents=3, stride=stride, offset=offset_n)
        self.top_vao.add_vbo(2, self.top_data, ncomponents=2, stride=stride, offset=offset_t)

        # setup EBO for drawing cylinder's side, bottom and top
        self.side_vao.add_ebo(self.side_indices)
        self.bottom_vao.add_ebo(self.bottom_indices)
        self.top_vao.add_ebo(self.top_indices)

        # setup textures
        self.uma.setup_texture("texture1", "./image/thuymac.jpeg")
        self.uma.setup_texture("texture2", "./image/beauty.jpeg")
        self.uma.setup_texture("bottom_texture", "./image/tieuvi.jpeg")
        self.uma.setup_texture("top_texture", "./image/thuylinh.jpeg")

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

        K_materials_2 = np.array([
            [0.1, 0.7, 0.8],  # diffuse
            [0.1, 0.7, 0.8],  # specular
            [0.1, 0.7, 0.8]  # ambient
        ], dtype=np.float32)

        shininess = 100.0
        phong_factor = 0.2  # blending factor for phong shading and texture

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1f(phong_factor, 'phong_factor')
        return self

    def draw(self, projection, view, model):
        trans_mat = T.translate(0, 0, -3)

        modelview = trans_mat @ view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(self.selected_texture, 'selected_texture')

        self.bottom_vao.activate()
        self.uma.upload_uniform_scalar1i(0, 'face')  # bottom = 0
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, self.bottom_indices.shape[0], GL.GL_UNSIGNED_INT, None)

        self.top_vao.activate()
        self.uma.upload_uniform_scalar1i(1, 'face')  # top = 1
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, self.top_indices.shape[0], GL.GL_UNSIGNED_INT, None)

        self.side_vao.activate()
        self.uma.upload_uniform_scalar1i(2, 'face')  # side=2
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.side_indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2
