## taken from https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py

import math
import numpy as np
import cv2

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """Compute region similarity as the Jaccard Index.

    Args:
        annotation   (ndarray): binary annotation   map. Shape: [n_frames,H,W] or [H,W]
        segmentation (ndarray): binary segmentation map. The same shape as `annotation`.
        void_pixels  (ndarray): optional mask with void pixels. The same shape as void_pixels.

    Return:
        jaccard (float | ndarray): region similarity. Shape: [n_frames] or scalar.
    """
    assert (
        annotation.shape == segmentation.shape
    ), f"Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match."
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert (
            annotation.shape == void_pixels.shape
        ), f"Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match."
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :]
            f_res[frame_id] = f_measure(
                segmentation[frame_id, :, :],
                annotation[frame_id, :, :],
                void_pixels_frame,
                bound_th=bound_th,
            )
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(
            f"db_eval_boundary does not support tensors with {annotation.ndim} dimensions"
        )
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = (
        bound_th
        if bound_th >= 1
        else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    )

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def compute_size_boundry_centroid(binary_mask):
    is_empty = not np.any(binary_mask)
    H, W = binary_mask.shape
    if is_empty:
        return (0, 0), (H//2, W//2), (W//2, W//2, H//2, H//2)
    else:
        y, x = np.where(binary_mask == True)
        left_boundary = np.min(x)
        right_boundary = np.max(x)
        top_boundary = np.min(y)
        bottom_boundary = np.max(y)

        centroid_x = int(left_boundary + right_boundary) // 2
        centroid_y = int(top_boundary + bottom_boundary) // 2

        width, height = right_boundary - left_boundary + 1, bottom_boundary - top_boundary + 1

    return (width, height), (centroid_x, centroid_y), (left_boundary, right_boundary, top_boundary, bottom_boundary)

def crop_mask(mask1, mask2):
    """
    crop a pair of masks according to the size of the larger mask
    """
    assert (mask1.shape == mask2.shape
    ), f"Annotation({mask1.shape}) and segmentation:{mask2.shape} dimensions do not match."

    mask1 = np.pad(mask1, ((mask1.shape[0], mask1.shape[0]), (mask1.shape[0], mask1.shape[0])), mode='constant', constant_values=False)
    mask2 = np.pad(mask2, ((mask2.shape[0], mask2.shape[0]), (mask2.shape[0], mask2.shape[0])), mode='constant', constant_values=False)

    size_1, centroid_1, boundary_1 = compute_size_boundry_centroid(mask1)
    size_2, centroid_2, boundary_2 = compute_size_boundry_centroid(mask2)

    width, height = max(size_1[0], size_2[0]), max(size_1[1], size_2[1])
    # print(f"Crop Width: {width}, Crop Height: {height}")

    compact_mask_1 = mask1[centroid_1[1] - height//2:centroid_1[1] + height//2 + 1, centroid_1[0] - width//2:centroid_1[0] + width//2 + 1]
    compact_mask_2 = mask2[centroid_2[1] - height//2:centroid_2[1] + height//2 + 1, centroid_2[0] - width//2:centroid_2[0] + width//2 + 1]

    return (size_1, size_2), (centroid_1, centroid_2), (compact_mask_1, compact_mask_2)

def getMidDist(gt_mask, pred_mask):

    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        gt_bigc = max(gt_contours, key = cv2.contourArea)
        pred_bigc = max(pred_contours, key = cv2.contourArea)
    except:
        return -1

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    return np.linalg.norm(gt_mid - pred_mid)

def getMidDistNorm(gt_mask, pred_mask):
    H, W = gt_mask.shape[:2]
    mdist = getMidDist(gt_mask, pred_mask)
    return mdist / np.sqrt(H**2 + W**2)

def getMidBinning(gt_mask, pred_mask, bin_size=5):

    H, W = gt_mask.shape
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        gt_bigc = max(gt_contours, key = cv2.contourArea)
        pred_bigc = max(pred_contours, key = cv2.contourArea)
    except:
        return -1

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    gt_x, gt_y = gt_mid.round()
    pred_x, pred_y = pred_mid.round()
    
    gt_bin_x, gt_bin_y = gt_x // bin_size, gt_y // bin_size
    pred_bin_x, pred_bin_y = pred_x // bin_size, pred_y // bin_size

    return (gt_bin_x == pred_bin_x) and (gt_bin_y == pred_bin_y)