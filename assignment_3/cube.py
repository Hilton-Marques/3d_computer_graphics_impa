import torch
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes


# Vertex coordinates for a level 0 cube.
_cube_verts0 = [
    [-0.50, 0.50, 0.50],
    [-0.50, -0.50, 0.50],
    [0.50, -0.50, 0.50],
    [0.50, 0.50, 0.50],

    [-0.50, 0.50, -0.50],
    [-0.50, -0.50, -0.50],
    [0.50, -0.50, -0.50],
    [0.50, 0.50, -0.50]
]


# Faces for level 0 cube
_cube_faces0 = [
    [0, 1, 2],
    [2, 3, 0],
    [7, 6, 5],
    [4, 7, 5],
    [6, 3, 2],
    [3, 6, 7],
    [4, 5, 0],
    [0, 5, 1],
    [3, 4, 0],
    [4, 3, 7],
    [2, 1, 5],
    [5, 6, 2],
]

def cube(level=0, device = None):
    """
    Create verts and faces for a unit cube, with all faces oriented
    consistently.
    Args:
        level: integer specifying the number of iterations for subdivision
                of the mesh faces. Each additional level will result in four new
                faces per face.
        device: A torch.device object on which the outputs will be allocated.
    Returns:
        Meshes object with verts and faces.
    """
    if device is None:
        device = torch.device("cpu")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level == 0:
        verts = torch.tensor(_cube_verts0, dtype=torch.float32, device=device)
        faces = torch.tensor(_cube_faces0, dtype=torch.int64, device=device)
    else:
        mesh = cube(level - 1, device)
        subdivide = SubdivideMeshes()
        mesh = subdivide(mesh)
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
        
    return Meshes(verts=[verts], faces=[faces])