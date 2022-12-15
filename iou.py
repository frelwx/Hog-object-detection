import torch
def iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 1] * wh[:, :, 0]
    union = area1[:, None] + area2 - inter
    return inter / union
def nms(boxes, scores, threshold):
    idx = torch.argsort(scores, descending=True)
    keep = []
    while(len(idx) > 0):
        now_best = idx[0]
        keep.append(now_best)
        idx = idx[1:]
        iou_with_left = iou(boxes[now_best].unsqueeze(dim=0), boxes[idx])[0]
        idx = idx[iou_with_left < threshold]
    return keep
if __name__ == '__main__':
    import random
    random.seed(0)
    boxes1 = torch.tensor([[ 72., 194., 218., 822.]])
    boxes2 = torch.tensor([[  0., 109., 220., 833.]])
    print(iou(boxes1, boxes2))
    # boxes1 = torch.zeros(3, 4)
    # for i in boxes1:
    #     i[0] = random.randint(0, 10)
    #     i[1] = random.randint(11, 20)
    #     i[2] = random.randint(21, 30)
    #     i[3] = random.randint(31, 40)
    # boxes2 = torch.zeros(2, 4)
    # for i in boxes2:
    #     i[0] = random.randint(0, 10)
    #     i[1] = random.randint(11, 20)
    #     i[2] = random.randint(21, 30)
    #     i[3] = random.randint(31, 40)
    # print(boxes1)
    # print(boxes2)
    # print(iou(boxes1, boxes2)[0])
    # print(iou(boxes1, boxes2)[1])
    # print(iou(boxes1, boxes2)[2])
    # box =  torch.tensor([[2,3.1,7,5],[3,4,8,4.8],[4,4,5.6,7],[0.1,0,8,1]]) 
    # score = torch.tensor([0.5, 0.3, 0.2, 0.4])
    
    # output = nms(boxes=box, scores=score, threshold=0.3)

    # print(output)
