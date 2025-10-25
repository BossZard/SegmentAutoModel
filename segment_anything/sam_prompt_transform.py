import torch
import numpy as np
import random

def transform_point_for_sam(points):
    # points shape: B, pos_point + neg_point, x+y+c
    # return: points coordinates: B, pos_point + neg_point, x+y
    # return: points labels: B, pos_point + neg_point, c
    # print('points shape:', points.shape)
    batch_size, points_num = points.shape[0], points.shape[1] // 2
    pos_points_arr = points[:, :points_num, :2]
    neg_points_arr = points[:, points_num:, :2]

    invalid_points = points[:,:,0] < 0
    pos_points_label = torch.ones_like(pos_points_arr[:, :, 0])
    neg_points_label = torch.zeros_like(neg_points_arr[:, :, 0])

    points_coords = torch.cat((pos_points_arr, neg_points_arr), dim=1)
    points_labels = torch.cat((pos_points_label, neg_points_label), dim=1)
    points_labels[invalid_points] = -1
    # print(points_coords, '\n', points_labels)

    return points_coords, points_labels

def transform_mask_for_sam(mask, new_size = (256, 256)):
    mask = torch.nn.functional.interpolate(
                mask, size=new_size, mode='bicubic', align_corners=False)

    return mask


def generate_bboxes_by_mask(high_res_masks, perturbation_range=20, reduce_scale=False):
    batch_size = high_res_masks.shape[0]
    bboxes = []
    for bs in range(batch_size):
        cur_mask = high_res_masks[bs]
        cur_bbox = get_bbox_by_single_mask(cur_mask, perturbation_range, reduce_scale)
        bboxes.append(cur_bbox)
    bboxes = torch.stack(bboxes, 0)

    return bboxes

def generate_anchor_by_prev_or_next_slice_mask(prev_mask, next_mask, orig_gt_mask, perturbation_range=20, reduce_scale=False):
    batch_size = prev_mask.shape[0]
    bboxes = []
    for bs in range(batch_size):
        cur_prev_mask = prev_mask[bs]
        cur_next_mask = next_mask[bs]
        cur_gt_mask = orig_gt_mask[bs]
        if random.random() >= 0.5:
            if torch.sum(cur_prev_mask)>0:
                cur_mask = cur_prev_mask
            elif torch.sum(cur_next_mask)>0:
                cur_mask = cur_next_mask
            else:
                cur_mask = cur_gt_mask
        else:
            if torch.sum(cur_next_mask)>0:
                cur_mask = cur_next_mask
            elif torch.sum(cur_prev_mask)>0:
                cur_mask = cur_prev_mask
            else:
                cur_mask = cur_gt_mask

        cur_bbox = get_bbox_by_single_mask(cur_mask, perturbation_range, reduce_scale)
        bboxes.append(cur_bbox)
    bboxes = torch.stack(bboxes, 0)

    return bboxes


def get_bbox_by_single_mask(gt2D, perturbation_range, reduce_scale):
    gt2D = gt2D.clone().squeeze().cpu().numpy()
    y_indices, x_indices = np.where(gt2D > 0)
    try:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
    except:
        return torch.tensor([-999, -999, -999, -999])
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    if perturbation_range==0:
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)
    else:
        if reduce_scale:
            x_left_range = -1 * min(20, (x_max - x_min) // 5)
            y_left_range = -1 * min(20, (y_max - y_min) // 5)
        else:
            x_left_range, y_left_range = 0, 0
        x_min = max(0, x_min - np.random.randint(x_left_range, perturbation_range))
        x_max = min(W, x_max + np.random.randint(x_left_range, perturbation_range))
        y_min = max(0, y_min - np.random.randint(y_left_range, perturbation_range))
        y_max = min(H, y_max + np.random.randint(y_left_range, perturbation_range))

    bboxes = torch.tensor([x_min, y_min, x_max, y_max])

    return bboxes
