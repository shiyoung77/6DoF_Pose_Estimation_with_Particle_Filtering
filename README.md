# 6DoF Pose Estimation with Particle Filter

## Introduction
This is a simple tool for 6DoF pose estimation of real-life objects, such as the ones from [YCB dataset](https://www.ycbbenchmarks.com/). This tool is based on a variant of the standard particle filtering algorithm. Given a single depth image, camera parameters, and the target object model, this tool is able to output the object 6DoF pose in the camera frame. It could be used to generate ground truth pose/segmentation of objects for your own object dataset.

## Installation
This tool is a fully written in Python and there is no need to explicitly compile anything. Nevertheless, you may still need to install a few third party python libraries such as: **[pycuda](https://documen.tician.de/pycuda/), [opencv-python](https://pypi.org/project/opencv-python/), [pyrender](https://pyrender.readthedocs.io/en/latest/), [trimesh](https://trimsh.org/), [mesh-to-sdf](https://github.com/marian42/mesh_to_sdf), [open3d](http://www.open3d.org/), [numba](https://numba.pydata.org/), [scikit-image](https://scikit-image.org/docs/stable/install.html)**. These libraries can be simply installed with `pip install {library-name}`. No specific version of these libraries are required. Please check their websites if there is any installation related issue.

## Example
An example can be found at `example.py` given the data in the `sample_data` folder. In this example, a window will pop up to let you crop a region of interest (a 2D bbox), and then pose estimation will be performed and a 2D segment of the object will be visualized. Note that the object mesh model (.obj) should be converted to the TSDF format before passing to this tool. A script to do this conversion could be found at `pf_pose_estimation/preprocess.py`. You can also preprocess your depth image if you have some prior knowledge, such as background removal using a segmentation model, to ensure the best performance.

## Citation
This tool is a byproduct of the research paper, **"Online Object Model Reconstruction and Reuse for Lifelong Improvement of Robot Manipulation"**. Please cite this paper if you find this tool useful in your own project. Note that the implementation of this tool is slightly different from the one proposed in this paper as some of the task specific parts, such as table/robot arm filtering and false pose rejection, are removed to make this tool as general as possible.

Part of the code is inspired by the [tsdf-fusion-python](https://github.com/andyzeng/tsdf-fusion-python), please also consider citing their paper.

```
@article{lu2021online,
title={Online Object Model Reconstruction and Reuse for Lifelong Improvement of Robot Manipulation},
author={Lu, Shiyang and Wang, Rui and Miao, Yinglong and Mitash, Chaitanya and Bekris, Kostas},
journal={arXiv preprint arXiv:2109.13910},
year={2021}
}
```
