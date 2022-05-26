import torch
def compute_iou(mask_gt,mask_predict):
    #mask_gt: 128-128
    #mask_predict: 128-128
    #compute IoU
    intersection = torch.logical_and(mask_gt,mask_predict)
    union = torch.logical_or(mask_gt,mask_predict)
    iou_score = intersection.sum()/union.sum()
    return iou_score

def compute_msc(mask_gt,mask_predict):
    #mask_gt: 128-128-4
    #mask_predict: 128-128-4
    msc_sum = 0.
    msc_count = 0.
    for i in range(mask_gt.shape[2]):
        mask_gt_i = mask_gt[:,:,i]
        if mask_gt_i.sum() == 0.:
            continue
        max_iou = -1000.
        for k in range(mask_predict.shape[2]):
            mask_predict_k = mask_predict[:,:,k]
            iou_i_k = compute_iou(mask_gt_i,mask_predict_k)
            max_iou = iou_i_k if iou_i_k > max_iou else max_iou
        msc_sum = msc_sum + max_iou
        msc_count+=1
    msc = msc_sum/msc_count
    return msc
