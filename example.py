import numpy as np
import cv2
import trimesh
import pyrender

from pf_pose_estimation.tsdf_lib import TSDFVolume
from pf_pose_estimation.particle_filter import ParticleFilter


if __name__ == '__main__':
    # camera info
    fx, fy = 459.906, 460.156
    cx, cy = 347.191, 256.039
    cam_intr = np.array([
        [fx, 0, cx], 
        [0, fy, cy],
        [0, 0, 1]
    ])
    depth_scale = 4000

    color_im_path = 'sample_data/color.png'
    depth_im_path = 'sample_data/depth.png'
    color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
    H, W = depth_im.shape

    obj_tsdf_path = 'sample_data/006_mustard_bottle/tsdf.npz'
    obj_tsdf_vol = TSDFVolume.load(obj_tsdf_path)
    pf = ParticleFilter(obj_tsdf_vol, num_particles=2000)
    pose, inlier_ratio = pf.estimate(color_im, depth_im, cam_intr, mask=None, num_iters=100, visualize=True)

    # =========================== test pose using original mesh file ===========================
    mesh_path = 'sample_data/006_mustard_bottle/textured.obj'
    fuze_trimesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
    camera_node = pyrender.Node(name='cam', camera=camera, matrix=np.eye(4))
    light_node = pyrender.Node(name='light', light=light, matrix=np.eye(4))
    mesh_node = pyrender.Node(name='mesh', mesh=mesh, matrix=np.eye(4))
    scene = pyrender.Scene(nodes=[camera_node, light_node, mesh_node])

    m = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    scene.set_pose(mesh_node, m @ pose)

    renderer = pyrender.OffscreenRenderer(W, H)
    color, depth = renderer.render(scene)
    cv2.imshow("depth", depth)
    cv2.waitKey(0)
