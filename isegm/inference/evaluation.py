from time import time

import numpy as np
import torch
import cv2

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.utils.feat_vis import save_image


try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

from medpy import metric


def evaluate_dataset(dataset, predictor, es_analysis, **kwargs):
    all_ious = []
    all_scribbles = []
    all_points = []

    all_dices = []
    all_hd95s = []
    all_assds = []

    all_times_per_image = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):

        sample = dataset.get_sample(index)

        start_spi = time()
        _, sample_ious, _ , sample_scribbles, sample_points, sample_dices, sample_hd95s, sample_assds, exception = evaluate_sample(sample.image, sample.gt_mask, sample._encoded_boundary,  
                                            predictor, sample_id=index, es_analysis=es_analysis, **kwargs)
        end_spi = time()
        if exception:
            continue
        all_times_per_image.append(end_spi - start_spi)
        if es_analysis:
            all_scribbles.append(sample_scribbles)
            all_points.append(sample_points)
        
        all_ious.append(sample_ious)
        all_dices.append(sample_dices)
        # all_hd95s.append(sample_hd95s)
        all_assds.append(sample_assds)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time, all_scribbles, all_points, all_dices, all_hd95s, all_assds, all_times_per_image


def evaluate_sample(image, gt_mask, _encoded_boundary, 
                    predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, es_analysis=False):
    clicker = Clicker(gt_mask=gt_mask, gt_boundary=_encoded_boundary) 
    pred_mask = np.zeros_like(gt_mask)

    edges = np.zeros_like(np.empty((2,gt_mask.shape[0],gt_mask.shape[1])),dtype=np.int32)
    ious_list = []
    scribbles_list = []
    points_list = []

    dice_list = []
    hd95_list = []
    assd_list = []

    with torch.no_grad():
        predictor.set_input_image(image)
        exception = False
        for click_indx in range(max_clicks):
            # click_indx is interaction round
            if es_analysis and (click_indx >= 10):
                edges = clicker.make_next_click(pred_mask, edges)
                edges = edges.astype(int)
                pred_probs = predictor.get_prediction(clicker, edges)
            else:
                clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, edges, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            dice = metric.binary.dc(pred_mask, gt_mask)
            try:
                # hd95 = metric.binary.hd95(pred_mask, gt_mask)
                assd = metric.binary.assd(pred_mask, gt_mask)
            except Exception:
                exception = True
                print('Raise a exception, image id: {}'.format(sample_id + 1))
                break
            ious_list.append(iou)
            if es_analysis:
                scribbles_list.append((cv2.countNonZero(edges[0])+cv2.countNonZero(edges[1]))/cv2.countNonZero(_encoded_boundary))
                points_list.append(cv2.countNonZero(edges[0])+cv2.countNonZero(edges[1]))

            dice_list.append(dice)
            # hd95_list.append(hd95)
            assd_list.append(assd)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs, scribbles_list, points_list, \
            np.array(dice_list, dtype=np.float32), np.array(hd95_list, dtype=np.float32), np.array(assd_list, dtype=np.float32), exception
