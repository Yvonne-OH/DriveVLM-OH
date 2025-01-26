import numpy as np

def IOU (boxA: np.array, boxB: np.array) -> float:

    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])

    inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1) # 计算交集面积

    area_boxA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_boxB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / float(area_boxA + area_boxB - inter_area) # 计算IOU

    return iou


def non_max_suppression(boxes: np.array, scores: np.array, iou_threshold=0.5):
    # 根据分数从高到低对框进行排序
    idxs = np.argsort(scores)[::-1]
    keep_boxes = []

    while len(idxs) > 0:
        # 取出当前分数最高的框的索引
        current_idx = idxs[0]
        current_box = boxes[current_idx]
        keep_boxes.append(current_box)

        # 计算当前框与其余框的 IOU
        ious = np.array([IOU(current_box, boxes[i]) for i in idxs[1:]])

        # 保留那些 IOU 小于阈值的框
        idxs = idxs[1:][ious < iou_threshold] # [ True False  True]

    return keep_boxes

def multi_class_NMS (boxes: np.array, scores: np.array, labels: np.array, iou_threshold=0.5):

    unique_labels = np.unique(labels)
    final_boxes = []


    for label in unique_labels:
        class_idxs = np.where(labels == label)[0]
        class_boxes = boxes[class_idxs]
        class_scores = scores[class_idxs]
        nms_boxes = non_max_suppression(class_boxes, class_scores, iou_threshold=iou_threshold)
        final_boxes.extend(nms_boxes)

    return final_boxes



# 示例用法
if __name__ == "__main__":
    # 单类别应用NMS
    # np.array()  创建numpy数组
    boxes = np.array([[10, 10, 40, 40], [11, 12, 43, 43], [9, 9, 39, 38], [10, 12, 43, 43]])  # [xmin, ymin, xmax, ymax]
    scores = np.array([0.9, 0.8, 0.7,0.95])  # 每个框的置信度
    iou_thresh = 0.9  # iou阈值

    # 应用NMS
    indices_to_keep = non_max_suppression(boxes, scores, iou_threshold=iou_thresh)
    print("保留的边界框索引:", indices_to_keep)

    # 示例边界框坐标、分数和类别标签
    boxes = np.array([[100, 100, 210, 210], [105, 110, 215, 220], [250, 250, 330, 330], [255, 260, 340, 340]])
    scores = np.array([0.9, 0.85, 0.88, 0.87])
    labels = np.array([0, 0, 1, 1])  # 假设0为车辆，1为行人

    result_boxes = multi_class_NMS(boxes, scores, labels)
    print("Filtered boxes:", result_boxes)


