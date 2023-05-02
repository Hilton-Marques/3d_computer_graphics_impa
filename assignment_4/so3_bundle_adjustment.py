# -*- coding: utf-8 -*-
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
        os.system("pip install fvcore iopath")
        os.system("pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html")
    else:
        # We try to install PyTorch3D from source.
        os.system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")

# imports
import matplotlib.pyplot as plt
import torch
import pytorch3d.transforms as t
from torch.optim.optimizer import Optimizer, required

from pytorch3d.transforms.so3 import (
    so3_exponential_map,
    so3_relative_angle,
)

from pytorch3d.renderer.cameras import (
    PerspectiveCameras,
)

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings, PointLights, MeshRenderer, 
    MeshRasterizer, SoftPhongShader
)

# add path for demo utils
import sys
import os
sys.path.append(os.path.abspath(''))

# set for reproducibility
torch.manual_seed(777)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

"""If using **Google Colab**, fetch the utils file for plotting the camera scene, and the ground truth camera positions:"""

#os.system("wget https://raw.githubusercontent.com/hallpaz/3dsystems23/master/scripts/camera_visualization.py")
#os.system("wget https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/docs/tutorials/utils/plot_image_grid.py")

from camera_visualization import plot_camera_scene
from plot_image_grid import image_grid

os.system("mkdir data")
os.system("wget -P data https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/docs/tutorials/data/camera_graph.pth")

"""OR if running **locally** uncomment and run the following cell:"""

# from utils import plot_camera_scene
# from utils import image_grid

"""## 1. Set up Cameras and load ground truth positions"""

# load the SE3 graph of relative/absolute camera positions
camera_graph_file = './data/camera_graph.pth'
import scipy.io

def concatenate_rotation_translation(R,T):
    N = R.size(0)
    RT = torch.cat((R, T.unsqueeze(2)), dim = 2)
    B = torch.tensor([[0,0,0,1]]).unsqueeze(0).repeat(N,1,1)
    return torch.cat((RT,B), dim = 1).permute(1,2,0)

def toMatlab(camera_graph_file):
    (R_absolute_gt, T_absolute_gt), (R_relative, T_relative), relative_edges = torch.load(camera_graph_file)    
    P_gt = concatenate_rotation_translation(R_absolute_gt, T_absolute_gt)
    P_rel = concatenate_rotation_translation(R_relative, T_relative)
    filename = 'camera_graph.mat'
    data_dict = {
    'P_gt': P_gt.numpy(),
    'P_rel': P_rel.numpy(),
    'graph_edge': relative_edges.numpy()}
    scipy.io.savemat(filename, data_dict, appendmat=True)

#toMatlab(camera_graph_file)    

(R_absolute_gt, T_absolute_gt), \
    (R_relative, T_relative), \
    relative_edges = \
        torch.load(camera_graph_file)

# create the relative cameras
cameras_relative = PerspectiveCameras(
    R = R_relative.to(device),
    T = T_relative.to(device),
    device = device,
)

# create the absolute ground truth cameras
cameras_absolute_gt = PerspectiveCameras(
    R = R_absolute_gt.to(device),
    T = T_absolute_gt.to(device),
    device = device,
)

# the number of absolute camera positions
N = R_absolute_gt.shape[0]

"""1.1 Check the ground truth values for rotation and translation of the first camera $g_0$. Do they look like measured values or arbitrary ones? Why do you think this decision was taken?"""

##############################################################################
# Code and explanation for 1.1
##############################################################################

"""## 2. Define optimization functions

### Relative cameras and camera distance
We now define two functions crucial for the optimization.

**`calc_camera_distance`** compares a pair of cameras. This function is important as it defines the loss that we are minimizing. The method utilizes the `so3_relative_angle` function from the SO3 API.

**`get_relative_camera`** computes the parameters of a relative camera that maps between a pair of absolute cameras. Here we utilize the `compose` and `inverse` class methods from the PyTorch3D Transforms API.
"""

def calc_camera_distance(cam_1, cam_2):
    """
    Calculates the divergence of a batch of pairs of cameras cam_1, cam_2.
    The distance is composed of the cosine of the relative angle between 
    the rotation components of the camera extrinsics and the L2 distance
    between the translation vectors.
    """
    # rotation distance
    # trans_x = cam_1.get_world_to_view_transform()
    # trans_exact = cam_2.get_world_to_view_transform()
    # e = trans_x.compose(trans_exact.inverse())
    # e_log = e.get_se3_log()
    # err = torch.sum(torch.sum(e_log * e_log, dim=-1))
    
    
    R_distance = (1.-so3_relative_angle(cam_1.R, cam_2.R, cos_angle=True)).mean()
    #translation distance
    T_distance = ((cam_1.T - cam_2.T)**2).sum(1).mean()
    # the final distance is the sum
    err = R_distance + T_distance
    return err

def get_relative_camera(cams, edges):
    """
    For each pair of indices (i,j) in "edges" generate a camera
    that maps from the coordinates of the camera cams[i] to 
    the coordinates of the camera cams[j]
    """

    # first generate the world-to-view Transform3d objects of each 
    # camera pair (i, j) according to the edges argument
    trans_i, trans_j = [ PerspectiveCameras(  R = cams.R[edges[:, i]],
            T = cams.T[edges[:, i]],
            device = device,
        ).get_world_to_view_transform()
         for i in (0, 1)
    ]
    
    # compose the relative transformation as g_i^{-1} g_j
    a = trans_i.inverse()
    #trans_rel = trans_j.compose(trans_i.inverse())
    trans_rel = trans_i.inverse().compose(trans_j)
    
    
    
    # generate a camera from the relative transform
    matrix_rel = trans_rel.get_matrix()
    cams_relative = PerspectiveCameras(
                        R = matrix_rel[:, :3, :3],
                        T = matrix_rel[:, 3, :3],
                        device = device,
                    )
    return cams_relative

def cat_in_place(t1,t2):
    m = t1.shape[0]
    n1 = t1.shape[1]
    n2 = t2.shape[1]    
    t = torch.zeros((m, n1 + n2))
    t[:, 0:n1] = t1
    t[:, n1:] = t2
    return t
    
    

def getJacobianRightInv(v):
    eps = 1e-6
    h = eps*torch.eye(6,6, dtype=torch.double )
    N = v.size(0)
    v = v.double()
    res = torch.zeros(N,6,6, dtype=torch.double)
    P = t.se3.se3_exp_map(v, eps = eps)
    for j in range(6):
        h_i  = h[j,:].unsqueeze(0)      
        M = t.se3.se3_exp_map(h_i, eps = eps).expand(N, -1, -1)
        Y = torch.bmm(M, P)
        Y_inv = Y.inverse()
        log = t.se3.se3_log_map(Y_inv,eps = eps)
        dH = -(log + v ) / eps
        res[:,:,j] = dH    
    return res
"""2.1 In this task, we are parameterizing the 3D rotation group - $SO(3)$ - using rotation matrices. This choice has some drawbacks, as we need to ensure our matrices are valid rotation matrices. Which other choice(s) could we have used to parameterize rotations? Would it be a better choice?"""

##############################################################################
# Explanation for 2.1
##############################################################################

"""## 3. Optimization
Finally, we start the optimization of the absolute cameras.

We use SGD with momentum and optimize over `log_R_absolute` and `T_absolute`. 

As mentioned earlier, `log_R_absolute` is the axis angle representation of the rotation part of our absolute cameras. We can obtain the 3x3 rotation matrix `R_absolute` that corresponds to `log_R_absolute` with:

`R_absolute = so3_exponential_map(log_R_absolute)`

"""

# initialize the absolute log-rotations/translations with random entries
log_R_absolute_init = torch.randn(N, 3, dtype=torch.float32, device=device)
T_absolute_init = torch.randn(N, 3, dtype=torch.float32, device=device)

# furthermore, we know that the first camera is a trivial one 
# (check exercise 1.1 above)
log_R_absolute_init[0, :] = 0.
T_absolute_init[0, :] = 0.

# instantiate a copy of the initialization of log_R / T
log_R_absolute = log_R_absolute_init.clone().detach()
log_R_absolute.requires_grad = True
T_absolute = T_absolute_init.clone().detach()
T_absolute.requires_grad = True

# the mask the specifies which cameras are going to be optimized
#     (since we know the first camera is already correct, 
#      we only optimize over the 2nd-to-last cameras)
camera_mask = torch.ones(N, 1, dtype=torch.float32, device=device)
camera_mask[0] = 0.

# init the optimizer
v = torch.cat((T_absolute, log_R_absolute), dim = 1).clone().detach()
v.requires_grad = True
#optimizer = torch.optim.SGD([log_R_absolute, T_absolute], lr=.1, momentum=0.9)
optimizer = torch.optim.SGD([v], lr=.1, momentum=0.9)

# run the optimization
n_iter = 500  # fix the number of iterations
loss = [0.0] * n_iter
for it in range(n_iter):
    # re-init the optimizer gradients
    optimizer.zero_grad()

    # compute the absolute camera rotations as 
    # an exponential map of the logarithms (=axis-angles)
    # of the absolute rotations
    #R_absolute = so3_exponential_map(log_R_absolute * camera_mask)
    poses = t.se3.se3_exp_map(v * camera_mask)
    R_absolute = poses[:,0:3, 0:3]
    T_absolute = poses[:,3, 0:3]
    # get the current absolute cameras
    cameras_absolute = PerspectiveCameras(
        R = R_absolute,
        T = T_absolute,
        device = device,
    )
    if it == -1:
        #first_pose = cameras_absolute.get_world_to_view_transform().get_matrix().permute(1,2,0)
        first_pose = concatenate_rotation_translation(R_absolute , T_absolute * camera_mask)
        scipy.io.savemat("first_poses.mat", {"first_pose" : first_pose.detach().numpy()})
        

    # compute the relative cameras as a compositon of the absolute cameras
    cameras_relative_composed = \
        get_relative_camera(cameras_absolute, relative_edges)

    # compare the composed cameras with the ground truth relative cameras
    # camera_distance corresponds to $d$ from the description
    camera_distance = \
        calc_camera_distance(cameras_relative_composed, cameras_relative)

    loss[it] = camera_distance.item()
    # our loss function is the camera_distance
    camera_distance.backward()
    
    # apply the gradients
    #apply modifications on SE3
    #x = torch.cat((T_absolute, log_R_absolute), dim = 1)
    #x = cat_in_place(T_absolute, log_R_absolute)
    #Jrinv = getJacobianRightInv(v)
    #dx = -0.1 * v.grad.unsqueeze(2).double()
    #v += torch.bmm(Jrinv ,dx).squeeze(2)
    optimizer.step()

    # plot and print status message
    if it % 200==0 or it==n_iter-1:
        status = 'iteration=%3d; camera_distance=%1.3e' % (it, camera_distance)
        #plot_camera_scene(cameras_absolute, cameras_absolute_gt, status)

fig, ax = plt.subplots(figsize=(10, 10))
ax.grid(True, which='both')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')

ax.plot(range(n_iter), loss, label="model")
plt.legend()
plt.show()
print('Optimization finished.')

"""3.1. Download the *cow mesh* and user the function `render_scene` (below) to render it using the cameras of the ground truth. Do the same using the initial values of the estimated cameras. 

*You don't need to understand how to set up a renderer now, we'll cover this later on the couser. For now, just focus on analyzing the results.*
"""

# download the cow mesh
os.system("mkdir -p data/cow_mesh")
os.system("wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj")
os.system("wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl")
os.system("wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png")

def render_scene(meshes, cameras, device):
  """
  Renders 3D meshes to a tensor of images.

  Args:
    meshes: a Meshes instance holding the meshes to be rendered
    cameras: a pytorch3D Cameras instance such as PerspectiveCameras
    device: a torch.device

  """
  if len(meshes) != len(cameras):
    meshes = meshes.extend(len(cameras))

  raster_settings = RasterizationSettings(
      image_size=512, 
      blur_radius=0.0, 
      faces_per_pixel=1, 
  )
  lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
  renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=cameras, 
          raster_settings=raster_settings
      ),
      shader=SoftPhongShader(
          device=device, 
          cameras=cameras,
          lights=lights
      )
  )
  return renderer(meshes).detach()

# you can visualize the images using the image_grid function:
# images = renderer(meshes, cameras, device)
# image_grid(images.cpu().numpy(), rows=4, cols=5, rgb=True)

# or you can choose a single image to check with matplotlib
# plt.figure(figsize=(10, 10))
# plt.imshow(images[0, ..., :3].cpu().numpy())
# plt.grid("off");
# plt.axis("off");

##############################################################################
# Code for 3.1
##############################################################################

"""3.2 Run the optimization loop and plot the  *loss vs iteration* graph. 

**[Extra] E.1: Can you do better (improve the approximation)?**

3.3 Render the images again, now using the ground truth cameras and the optimized cameras. Describe the results qualitatively.

**[Extra] E.2: Use another representation for rotation matrices to solve the bundle adjustment problem.**
"""

##############################################################################
# Code and explanation for 3.2 - 3.3 (and extras)
##############################################################################
