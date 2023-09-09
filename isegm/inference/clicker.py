from tkinter.messagebox import NO
import numpy as np
from copy import deepcopy
import cv2
from isegm.utils.feat_vis import save_image
import torch
import skimage


class Clicker(object): 
    def __init__(self, gt_mask=None, gt_boundary=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None
        self.gt_boundary = gt_boundary

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)
        self.cnt = 0
        self.boundaryPixelTotal = cv2.countNonZero(gt_boundary) 


    def make_next_click(self, pred_mask, edges, aux_edge):
        assert self.gt_mask is not None
        click, edges = self._get_next_click(pred_mask, edges, aux_edge)
        self.add_click(click)
        return edges

    def make_next_click(self, pred_mask, edges=None):
        assert self.gt_mask is not None
        click, updated_edges = self._get_next_click(pred_mask, edges)
        self.add_click(click)
        if edges is not None:
            return updated_edges

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, edges, padding=True):
        self.cnt += 1

        finetune = False
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)
 
        if edges is not None: 
            fn_region_label = skimage.measure.label(fn_mask) # 1、连通区域标记
            fp_region_label = skimage.measure.label(fp_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0) # 疑问：为什么第3个参数mask_size为0？ 
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist # 判断下一次点击是正例点击还是负例点击
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
        
        if edges is not None:
            region_label_mask = fn_region_label if is_positive else fp_region_label # 2、点击点所在的负例区域或者正例区域
            click_label = region_label_mask[coords_y[0], coords_x[0]] # 3、点击点的标记索引i
            region_label_mask_fore = (region_label_mask == click_label) # 4、索引i所在的错分区域，前景1，背景0
            region_label_mask_back = (region_label_mask != click_label)*5 # 索引i所在的错分区域，前景0，背景5
            label_mask = region_label_mask_fore + region_label_mask_back # 索引i所在的错分区域，前景1，背景5

            # 限制勾勒量 ablation 3
            radius_mask = np.ones_like(edges[0])
            radius_mask[coords_y[0],coords_x[0]] = 0
            radius_mask_dt = cv2.distanceTransform(radius_mask.astype(np.uint8), cv2.DIST_L2, 5)
            # ratio = 0.02
            # limit_dt = self.boundaryPixelTotal*ratio
            # default: no limit
            limit_dt = self.boundaryPixelTotal
            
            radius_mask_dt = (radius_mask_dt<limit_dt).astype(int)
            label_mask = (self.gt_boundary==label_mask).astype(int)
            label_mask = label_mask*radius_mask_dt
            if is_positive:
                edges[0] = edges[0] + label_mask
            else:
                edges[1] = edges[1] + label_mask

            edges = (edges > 0)
        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0])), edges

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool_)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None, finetune=False):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx
        self.finetune = finetune

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
