from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import glfw



def normal_of_face(A, B, C):
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)
    n = n/np.linalg.norm(n)
    return n

def normal_of_vert(n1, n2, n3):
    n = n1 + n2 + n3
    n = n / np.linalg.norm(n)
    return n


class Tetrahedron(object):
    def __init__(self, vert_shader, frag_shader):
        X = 1
        P = np.array(
            [
                [-3, 0, -1],  # A
                [+1, X, +4],  # B
                [+3, X, -3]  # C
            ],
            dtype=np.float32
        )
        A, B, C = P[0, :], P[1, :], P[2, :]
        nABC = normal_of_face(A, B, C)
        D = A + 5.0 * nABC

        self.vertices = np.concatenate([
            A.reshape(1, -1),
            B.reshape(1, -1),
            C.reshape(1, -1),
            D.reshape(1, -1),
        ], axis=0)

        self.indices = np.array(
            [0, 3, 1, 3, 2, 3, 0, 3, 3, 0, 0, 1, 2],
            dtype=np.int32
        )

        # faces: ACB, DAB, DBC, DCA
        nACB = normal_of_face(A, C, B)
        nDAB = normal_of_face(D, A, B)
        nDBC = normal_of_face(D, B, C)
        nDCA = normal_of_face(D, C, A)
        nA = normal_of_vert(nACB, nDCA, nDAB)
        nB = normal_of_vert(nACB, nDAB, nDBC)
        nC = normal_of_vert(nACB, nDCA, nDBC)
        nD = normal_of_vert(nDAB, nDBC, nDCA)

        self.normals = np.concatenate([
            nA.reshape(1, -1),
            nB.reshape(1, -1),
            nC.reshape(1, -1),
            nD.reshape(1, -1),
        ], axis=0)

        # colors: RGB format
        self.colors = np.array(
            [  # R    G    B
                [1.0, 0.0, 0.0],  # A
                [0.0, 1.0, 0.0],  # B
                [0.0, 0.0, 1.0],  # C
                [0.0, 0.0, 0.0]  # D
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
        #yrot_mat = T.rotate(T.vec(0, 1, 0), self.yrot_angle)
        #xrot_mat = T.rotate(T.vec(1, 0, 0), self.xrot_angle)
        trans_mat = T.translate(0, -2, -8)



        GL.glUseProgram(self.shader.render_idx)
        #modelview = view
        modelview = trans_mat @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

