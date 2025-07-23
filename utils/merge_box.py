import numpy as np

def box_iou(boxA, boxB):
    # box: [xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea)
    return iou
def convert_ocr_results(ocr_results):
    for ocr in ocr_results:
        quad = ocr["box"]
        text = ocr["text"]
        if isinstance(quad[0], list):
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
        else:
            xs = quad[0::2]
            ys = quad[1::2]
        box_ocr = [min(xs), min(ys), max(xs), max(ys)]
        ocr["box"] = box_ocr
        ocr["row_index"] = -1
        ocr["column_index"] = -1
        ocr["row_header"] = False
        ocr["spanning_cell"] = False
        ocr["cell"] = False
    return ocr_results

def convert_table_boxs(table_boxes):
    labels = table_boxes.get("labels", [])
    boxes = table_boxes.get("boxes", [])
    
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    boxes = boxes.tolist() if hasattr(boxes, "tolist") else boxes
    
    results = []
    
    column_boxes = [(box, label) for box, label in zip(boxes, labels) if label == 1]
    column_boxes = sorted(column_boxes, key=lambda x: x[0][0])
    for idx, (box, label) in enumerate(column_boxes):
        result = {
            "box": box,
            "label": label,
            "column_index": idx
        }
        results.append(result)
    row_boxes = [(box, label) for box, label in zip(boxes, labels) if label == 2]
    row_boxes = sorted(row_boxes, key=lambda x: x[0][1])
    for idx, (box, label) in enumerate(row_boxes):
        result = {
            "box": box,
            "label": label,
            "row_index": idx
        }
        results.append(result)
    for box, label in zip(boxes, labels):
        if label not in [1, 2]:
            result = {"box": box, "label": label}
            if label == 3:
                result["row_header"] = True
            elif label == 4:
                result["cell"] = True
            elif label == 5:
                result["spanning_cell"] = True
            results.append(result)
    
    return results

def text_not_in_table(ocr_results, table_boxes):
    labels = table_boxes.get("labels", [])
    boxes = table_boxes.get("boxes", [])
    
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    boxes = boxes.tolist() if hasattr(boxes, "tolist") else boxes
    table_boxes_list = [box for label, box in zip(labels, boxes) if label == 0]

    text_not_in_table = []
    for ocr in ocr_results:
        quad = ocr["box"]
        if isinstance(quad[0], list):
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
        else:
            xs = quad[0::2]
            ys = quad[1::2]
        box_ocr = [min(xs), min(ys), max(xs), max(ys)]
        is_inside = False
        for table_box in table_boxes_list:
            if box_iou(box_ocr, table_box) > 0.5:
                is_inside = True
                break
        if not is_inside:
            text_not_in_table.append(ocr["text"])
    return text_not_in_table


def merge_table_ocr(table_boxes, ocr_results):
    boxes = table_boxes.get("boxes", [])
    labels = table_boxes.get("labels", [])
    boxes = boxes.tolist()
    ocr_results = convert_ocr_results(ocr_results)
    table_boxes = convert_table_boxs(table_boxes)
    for ocr in ocr_results:
        box_ocr = ocr["box"]
        for i in range(1,6):
            max_iou = 0
            tag = ""
            if i == 1:
                tag = "column_index"
            elif i == 2:
                tag = "row_index"
            elif i == 3:
                tag = "row_header"
            elif i == 4:
                tag = "cell"
            elif i == 5:
                tag = "spanning_cell"
            index = -1
            for table_box in table_boxes:
                if table_box["label"] != i:
                    continue
                iou = box_iou(box_ocr, table_box["box"])
                if iou > max_iou:
                    max_iou = iou
                    index = table_box[tag]
            ocr[tag] = index
    ocr_results = remove_duplicates(ocr_results)
    return ocr_results

def remove_duplicates(data):
    unique = []
    seen = set()
    for item in data:
        key = (
            tuple(item["box"]),
            item.get("text", ""),
            item.get("row_index", -1),
            item.get("column_index", -1)
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
