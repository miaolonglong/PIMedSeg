from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

class BUSIDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='val', masks_dir_name='val_mask', boundarys_dir_name = 'val_boundary_aug',
                
                 **kwargs):
        super(BUSIDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name
        self._boundary_path = self.dataset_path / boundarys_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))] #源文件名列表
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}
        self._boundarys_paths = {x.stem: x for x in self._boundary_path.glob('*.*')}
 
    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index] #得到文件名 img.jpg
        image_path = str(self._images_path / image_name) #源文件路径
        mask_path = str(self._masks_paths[image_name.split('.')[0]]) #mask路径，根据源文件名查找mask
        boundary_path = str(self._boundarys_paths[image_name.split('.')[0]]) #mask路径，根据源文件名查找mask

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #opencv读的是BGR
        instances_mask = cv2.imread(mask_path)[:,:,0].astype(np.int32) #8位单通道图像，因此只有一个通道；将图片转成矩阵
        instances_mask[instances_mask > 0] = 1
        instances_ids = [1]
        instances_boundary = cv2.imread(boundary_path)[:,:,0].astype(np.int32) #8位单通道图像，因此只有一个通道；将图片转成矩阵
        instances_boundary[instances_boundary > 0] = 1

        return DSample(image, instances_mask, instances_boundary, objects_ids=instances_ids, sample_id=index)


