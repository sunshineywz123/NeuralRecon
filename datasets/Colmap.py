import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import json
# import OpenEXR
from pathlib import Path, PurePath
import cv2
import pycolmap
import glob
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
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split, img_files

class ColmapDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(ColmapDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

        self.reconstruction = pycolmap.Reconstruction("data/皮卡丘大占比对齐/tkl_model/")

        intrinsic = self.reconstruction.cameras[1].params
        self.cam_intr = np.eye(3)
        self.cam_intr[0][0] =  intrinsic[0]
        self.cam_intr[1][1] =  intrinsic[1]
        self.cam_intr[0][2] =  intrinsic[2]
        self.cam_intr[1][2] =  intrinsic[3]
        # import ipdb;ipdb.set_trace()
    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]


        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            image = self.reconstruction.images[vid+1]
            # load images
            img = self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'images', image.name))
            img_arr = np.asarray(img)
            img_bgr = img_arr[:, :, ::-1] 

            img_bgr = Image.fromarray(img_bgr)
            imgs.append(
                img_bgr
            )
            

            depth_im = cv2.imread( os.path.join(self.datapath, self.source_path, meta['scene'], 'depths', image.name),cv2.IMREAD_UNCHANGED)
            depth_im = depth_im.astype(np.float32)*0.3/255.0/255.0
            depth_im[depth_im > 3.0] = 0
            
            depth.append(
                depth_im
            )

            
            projection_matrix = np.eye(4)
            P=image.projection_matrix()
            projection_matrix[:3,:]=P

            intrinsics = self.cam_intr
            extrinsics = projection_matrix
            
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
