import torch
def _box_xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.as_tensor([cx, cy, w, h], device=boxes.device)
    return boxes

def _box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.as_tensor([x1, y1, x2, y2], device=boxes.device)
    return boxes

def get_shift_and_amplify_params(xywh, anchor_boxes):
    x, y, w, h = xywh
    x_a, y_a, w_a, h_a = anchor_boxes
    tx = (x - x_a) / w_a
    ty = (y - y_a) / h_a
    tw = torch.log(w / w_a)
    th = torch.log(h / h_a)
    t = torch.as_tensor([tx, ty, tw, th], device=anchor_boxes.device)
    return t
    
def shift_and_amplify_anchors(t, anchor_boxes):
    t_x, t_y, t_w, t_h = t
    x_a, y_a, w_a, h_a = anchor_boxes
    x = (t_x * w_a) + x_a
    y = (t_y * h_a) + y_a
    w = torch.exp(t_w) * w_a
    h = torch.exp(t_h) * h_a
    proposals = torch.as_tensor([x, y, w, h], device=anchor_boxes.device)
    return proposals