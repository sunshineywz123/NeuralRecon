import torch
import numpy as np
import os
import cv2

import json
from pathlib import Path, PurePath
import OpenEXR

def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)
    
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender_data(basedir, half_res=False, test_ratio=0.125):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    counts = [0]
    imgs = []
    img_files = []
    poses = []

    for frame in meta['frames']:
        fname = os.path.join(basedir, 'images', frame['file_path'].split('/')[-1] + '.png')
        img_files.append(fname)
        imgs.append(cv2.imread(fname))
        pose = np.array(frame['transform_matrix'])
        pose[:,1:3] *= -1
        # pose[2, 3] += 1.5
        poses.append(pose)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    counts.append(counts[-1] + imgs.shape[0])
    print(imgs.shape, poses.shape)

    n_images = len(imgs)
    freq_test = int(1/test_ratio)
    i_val = i_test = np.arange(0, n_images, freq_test)
    i_train = np.asarray(list(set(np.arange(n_images).tolist())-set(i_test.tolist())))
    i_split = [i_train, i_val, i_test]
    # print('TRAIN views are', i_train)
    # print('VAL views are', i_val)
    # print('TEST views are', i_test)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # import ipdb;ipdb.set_trace()
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    # poses[2, 3] += 1.5
    return imgs, poses, render_poses, [H, W, focal], i_split, img_files

class OmniObject3DDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, n_imgs, scene, data_path, max_depth, id_list=None):
        """
        Args:
        """
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list
        _,self.poses,_,[H, W, focal],_,_ = load_blender_data(Path(self.data_path+'/'+self.scene+'/render'), half_res=False)
        cam_intr = np.eye(3,3)
        self.cam_intr = cam_intr
        self.cam_intr[0][0] =  focal
        self.cam_intr[1][1] =  focal
        self.cam_intr[0][2] =  H/2
        self.cam_intr[1][2] =  W/2
        # translations = self.poses[:, :3, 3]

        # 计算平均旋转矩阵和平均平移向量

        # avg_translation = np.mean(translations, axis=0)
    
        # import ipdb;ipdb.set_trace()
    def __len__(self):
        return self.n_imgs
    

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]

        # cam_pose = np.linalg.inv(self.poses[id])
        cam_pose = self.poses[id]
        # Read depth image and camera pose
        # import ipdb;ipdb.set_trace()
        
        # 打开exr文件
        exr_file = OpenEXR.InputFile( os.path.join(self.data_path, self.scene, 'render/depths', 'r_'+str(id)+'_depth' +'.exr'))
        # 获取图像宽度、高度和通道数
        dw = exr_file.header()['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = exr_file.header()['channels']
        
        # 读取每个通道的数据
        data = {}
        for channel in channels:
            data[channel] = np.frombuffer(exr_file.channel(channel), dtype=np.float32)
            data[channel] = np.reshape(data[channel], (height, width))

        # 关闭exr文件
        exr_file.close()

        # 打印图像尺寸和通道数
        print(f"Image size: {width} x {height}")
        print("Channels:", channels)

        # 打印第一个像素的值
        for channel, values in data.items():
            print(f"{channel}: {values[0, 0]}")

        # depth_im = data['R']
        # depth_im = depth_im/1000  # depth is saved in 16-bit PNG in millimeters
        depth_im = 1/data['R']
        depth_im[depth_im > self.max_depth] = 0

        # Read RGB image
        color_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "render/images", 'r_'+str(id) + ".png")),
                                   cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)

        return cam_pose, depth_im, color_image
