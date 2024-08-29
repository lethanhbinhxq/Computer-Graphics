import numpy as np
import math
np.set_printoptions(formatter={'all':lambda x: "{:5.1f}".format(x)})

def generate_side(nsegments=50, height=1):
    angles = np.linspace(0, 2*math.pi, nsegments+1, endpoint=True)
    vx, vz = np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)
    vy_low, vy_high = np.zeros_like(vx), np.ones_like(vx)
    vertices = np.concatenate([vx, vy_low, vz, vx, height*vy_high, vz], axis=1)
    vertices = vertices.reshape(-1, 3)
    normals = np.concatenate([vx, vy_low, vz, vx, vy_low, vz], axis=1)
    normals = normals.reshape(-1, 3)
    tx = np.linspace(0, 1, nsegments+1, endpoint=True).reshape(-1, 1)
    ty_low, ty_high = np.ones_like(tx), np.zeros_like(tx)
    texcoords = np.concatenate([tx, ty_low, tx, ty_high], axis=1)
    texcoords = texcoords.reshape(-1, 2)
    vertex_attrib = np.concatenate([vertices, texcoords, normals], axis=1)
    return vertex_attrib.astype(np.float32)

def generate(nsegments=50, height=1):
    angles = np.linspace(0, 2*math.pi, nsegments+1, endpoint=True)
    vx, vz = np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)
    vy_low, vy_high = np.zeros_like(vx), np.ones_like(vx)
    vertices = np.concatenate([vx, vy_low, vz, vx, height*vy_high, vz], axis=1)
    vertices = vertices.reshape(-1, 3)
    normals = np.concatenate([vx, vy_low, vz, vx, vy_low, vz], axis=1)
    normals = normals.reshape(-1, 3)
    tx = np.linspace(0, 1, nsegments+1, endpoint=True).reshape(-1, 1)
    ty_low, ty_high = np.ones_like(tx), np.zeros_like(tx)
    texcoords = np.concatenate([tx, ty_low, tx, ty_high], axis=1)
    texcoords = texcoords.reshape(-1, 2)
    side_data = np.concatenate([vertices, texcoords, normals], axis=1).astype(np.float32)
    side_indices = np.arange(side_data.shape[0]).astype(np.int32)  # triangle strip

    bottom_center = np.array([0, 0, 0]).reshape(1, -1)
    bottom_border = np.concatenate([vx, vy_low, vz], axis=1)
    bottom_vertices = np.concatenate([bottom_center, bottom_border], axis=0)
    bottom_normals = np.array([0, -1, 0]).reshape(1, -1)
    bottom_normals = np.tile(bottom_normals, (bottom_vertices.shape[0], 1))
    bottom_center_tex = np.array([0, 0]).reshape(1, -1)
    bottom_texcoords = np.concatenate([vx, vz], axis=1)
    bottom_texcoords = np.concatenate([bottom_center_tex, bottom_texcoords], axis=0)
    bottom_texcoords = 0.5*bottom_texcoords + np.array([0.5, 0.5]).reshape(1, -1)
    bottom_data = np.concatenate([bottom_vertices, bottom_texcoords, bottom_normals], axis=1).astype(np.float32)
    bottom_indices = np.arange(bottom_data.shape[0]).astype(np.int32)  # triangle fan

    top_center = np.array([0, 1, 0]).reshape(1, -1)
    top_border = np.concatenate([vx, vy_high, vz], axis=1)
    top_vertices = np.concatenate([top_center, top_border], axis=0)
    top_normals = np.array([0, 1, 0]).reshape(1, -1)
    top_normals = np.tile(top_normals, (top_vertices.shape[0], 1))
    top_center_tex = np.array([0, 0]).reshape(1, -1)
    top_texcoords = np.concatenate([vx, vz], axis=1)
    top_texcoords = np.concatenate([top_center_tex, top_texcoords], axis=0)
    top_texcoords = 0.5 * top_texcoords + np.array([0.5, 0.5]).reshape(1, -1)
    top_data = np.concatenate([top_vertices, top_texcoords, top_normals], axis=1).astype(np.float32)
    top_indices = np.arange(top_data.shape[0]).astype(np.int32)  # triangle fan
    return side_data, side_indices, bottom_data, bottom_indices, top_data, top_indices

"""
side_data, side_indices, bottom_data, bottom_indices, top_data, top_indices = generate(nsegments=4, height=1)
print(side_data.shape)
print(bottom_data.shape)
print(top_data.shape)
print(top_data)
print(bottom_data)
#print(mesh)"""


