import cv2
import numpy as np
def draw_bounding_boxes(image_path, output):
    image = cv2.imread(image_path)
    for item in output:
        box = item['box']
        text = item['text']

        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = box[0]
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return image
def reconstruct_table(results):
    row_indices = [item['row_index'] for item in results if 'row_index' in item]
    col_indices = [item['column_index'] for item in results if 'column_index' in item]
    min_row = min(row_indices, default=0)
    max_row = max(row_indices, default=0)
    min_col = min(col_indices, default=0)
    max_col = max(col_indices, default=0)
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1

    table = [['' for _ in range(cols)] for _ in range(rows)]
    for item in results:
        row_index = item.get('row_index', -1) - min_row
        column_index = item.get('column_index', -1) - min_col
        text = item.get('text', '')
        if 0 <= row_index < rows and 0 <= column_index < cols:
            table[row_index][column_index] = text
    return table

def print_table(table):
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    for row in table:
        formatted_row = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(formatted_row)
    print()

