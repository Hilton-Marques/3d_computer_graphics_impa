import os
import sys
import torch
need_pytorch3d=False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d=True
if need_pytorch3d:
    if torch.__version__.startswith(("1.13.", "2.0.")) and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
        version_str="".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".",""),
            f"_pyt{pyt_version_str}"
        ])
        #!pip install fvcore iopath
        #!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
    else:
        # We try to install PyTorch3D from source.
        #!pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
        a = 1

"""## Data Structures"""

from pytorch3d.structures import Meshes

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

#!wget -P . https://raw.githubusercontent.com/hallpaz/3dsystems20/master/extensions_utils/cube.py

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

from cube import cube

refinedcube = cube(1, device=device)

verts_list = [torch.tensor(_cube_verts0, device=device), refinedcube.verts_list()[0]]
faces_list = [torch.tensor(_cube_faces0, dtype=torch.int64, device=device), refinedcube.faces_list()[0]]

mesh_batch = Meshes(verts=verts_list, faces=faces_list)

"""## Packed and Padded Tensors"""

# packed representation
verts_packed = mesh_batch.verts_packed()

# auxiliary tensors
mesh_to_vert_idx = mesh_batch.mesh_to_verts_packed_first_idx()
vert_to_mesh_idx = mesh_batch.verts_packed_to_mesh_idx()

# edges
edges = mesh_batch.edges_packed()

# face normals
face_normals = mesh_batch.faces_normals_packed()

verts_packed

mesh_batch.verts_padded()

"""## Input / Output"""

# !mkdir -p data
# !wget -P data https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj
# !wget -P data https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl
# !wget -P data https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png

from pytorch3d.io import load_obj

obj_file = "data/cow.obj"
verts, faces, aux = load_obj(obj_file)

faces = faces.verts_idx
normals = aux.normals
textures = aux.verts_uvs
materials = aux.material_colors
tex_maps = aux.texture_images

tex_maps

import matplotlib.pyplot as plt
from pytorch3d.renderer import Textures

plt.imshow(tex_maps['material_1'])

"""# 3D Transforms"""

from pytorch3d.transforms import Transform3d, Rotate, Translate

# example 1
T = Translate(torch.FloatTensor([[1.0, 2.0, 3.0]]), device=device)
R = Rotate(torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), device=device)
RT = Transform3d(device=device).compose(R, T)

RT.get_matrix()

# applying Transform
verts_transformed = RT.transform_points(mesh_batch.verts_packed())
verts_transformed

"""# Renderer"""

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader,
    Textures
)

R, T = look_at_view_transform(2.7, 10, 20)
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1, # sets the value of K
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras)
)

# Creating a texture for the mesh
white_tex = torch.ones_like(mesh_batch.verts_padded())
textures = Textures(verts_rgb=white_tex.to(device))
mesh_batch.textures = textures

images = renderer(mesh_batch, cameras=cameras)

def plot_side_by_side(images):
  n = images.shape[0]
  fig = plt.figure(figsize=(10, 10))
  for i in range(n):
    fig.add_subplot(1, n, i+1)
    plt.imshow(images[i, ..., :3].cpu().numpy())
    # plt.grid("off");
    # plt.axis("off");

#plot_side_by_side(images)

from math import radians, cos, sin

cos45 = cos(radians(45))
sin45 = sin(radians(45))
# applying a transform to the first mesh
SR = Transform3d(device=device).scale(1.0, 1.5, 1.0).rotate(
      R=torch.tensor([[cos45, -sin45, 0.0], [sin45, cos45, 0.0], [0.0, 0.0, 1.0]])
    )
verts0 = mesh_batch.verts_list()[0]
verts0 = SR.transform_points(verts0)
verts1 = mesh_batch.verts_list()[1]
mesh_batch2 = Meshes(verts=[verts0, verts1], faces=mesh_batch.faces_list(), textures=mesh_batch.textures)

#plot_side_by_side(renderer(mesh_batch2))

"""## Challenge

1. Change the texture of the mesh_batch so that each of the cubes is colored differently.

2. Experiment different transforms and compositions in terms of rotation, translation and scaling.
"""

# write your code below this cell
new_color = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float)
new_tex = torch.ones_like(mesh_batch.verts_padded())
new_tex[0,mesh_batch.verts_list()[0].shape[0]:,:] = new_color
new_tex[1,:,:] = new_color
textures_red = Textures(verts_rgb=new_tex.to(device))

cos30 = cos(radians(30))
sin30 = sin(radians(30))
SR = Transform3d(device=device).scale(1.0, 1.5, 1.0).rotate(
      R=torch.tensor([[cos30, -sin30, 0.0], [cos30, sin30, 0.0], [0.0, 0.0, 1.0]])
    )

verts1 = mesh_batch.verts_list()[1]
new_vertices = SR.transform_points(verts1)
mesh_batch_2 = Meshes(verts=[mesh_batch.verts_list()[0], new_vertices], faces=mesh_batch.faces_list(), textures=textures_red)
#mesh_batch2 = Meshes(verts=[verts0, verts1], faces=mesh_batch.faces_list(), textures=mesh_batch.textures)

images = renderer(mesh_batch_2, cameras=cameras)

plot_side_by_side(images)


"""## Implicit Modeling and *cubify*"""

from pytorch3d.ops import cubify
from pytorch3d.io import IO

x_axis = [-1, 1]
y_axis = [-1, 1]
z_axis = [-1, 1]
depth = 64
height = 64
width = 64

volume = torch.zeros([depth, height, width])

# some examples of surfaces defined implicitly
sphere = lambda x: x[0,:,:,:]*x[0,:,:,:] + x[1,:,:,:]*x[1,:,:,:] + x[2,:,:,:]*x[2,:,:,:]
torus = lambda x: torch.pow(1.0 - torch.sqrt(x[0,:,:,:]*x[0,:,:,:] + x[1,:,:,:]*x[1,:,:,:]),2) + x[2,:,:,:]*x[2,:,:,:] 




"""## Challenge

3. Can you substitute the ````for```` loops for vectorized operations using Numpy or PyTorch functions?

4. Can you make the cubified sphere look "rounded"? 

5. Train a neural network to learn a occupancy function for a 3D surface. Use the ```cubify``` method to generate a mesh, and visualize it.
"""

range = torch.linspace(-1,1,64)
X,Y,Z = torch.meshgrid(range,range,range)
points = torch.cat((X.unsqueeze(0),Y.unsqueeze(0),Z.unsqueeze(0)))

orbit = sphere(points)
mask = orbit < 1
volume[mask] = 1.0

cubified = cubify(volume.unsqueeze(0), 0.7)
IO().save_mesh(cubified, "cubified_mesh.obj")

#round the sphere
vertices = cubified.verts_packed()
vnorm = torch.linalg.vector_norm(vertices, dim=-1).unsqueeze(1)
verts = vertices / vnorm
new_sphere = Meshes([verts], cubified.faces_list())

IO().save_mesh(new_sphere, "new_sphere.obj")