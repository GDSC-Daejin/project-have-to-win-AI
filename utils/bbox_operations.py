# bbox_operations.py
from os import listdir
from os.path import isfile, join
from typing import Tuple

def adjust_bbox_coordinates(
        x: float,
        y: float,
        w: float,
        h: float
) -> Tuple[float, float, float, float]:
    
    # xywh를 xyxy로 변환
    x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h)

    # 좌표를 0~1 범위 내에 있는지 확인하고 벗어난 경우 조정
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)

    # xyxy를 xywh로 다시 변환
    x, y, w, h = xyxy_to_xywh(x1, y1, x2, y2)

    return x, y, w, h


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


def denormalize_xyxy(
        normalized_coordinates: Tuple[float, float, float, float],
        img_width: int,
        img_height: int
) -> Tuple[int, int, int, int]:
    
    x_min, y_min, x_max, y_max = normalized_coordinates

    denormalized_x_min = int(x_min * img_width)
    denormalized_y_min = int(y_min * img_height)
    denormalized_x_max = int(x_max * img_width)
    denormalized_y_max = int(y_max * img_height)

    return (denormalized_x_min, denormalized_y_min, denormalized_x_max, denormalized_y_max)


def process_yolo_label(folder_path: str, output_folder: str) -> None:

    # 폴더 내의 모든 파일 리스트
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith('.txt')]

    for file_name in files:
        # 이미지 파일 경로 및 YOLO 라벨 파일 경로
        label_path = join(folder_path, file_name)
        output_path = join(output_folder, file_name)

        # YOLO 라벨 로드
        with open(label_path, 'r', encoding='cp949') as label_file:
            lines = label_file.readlines()

        output_lines = []

        for line in lines:
            # YOLO 라벨 파싱
            label = line.strip().split()
            x, y, w, h = map(float, label[1:])  # 첫 번째 값은 클래스, 나머지는 바운딩 박스 좌표

            # 좌표 조정
            x, y, w, h = adjust_bbox_coordinates(x, y, w, h)

            # 정규화된 좌표를 출력 (파일에 저장)
            adjusted_line = f'{label[0]} {x} {y} {w} {h}'
            output_lines.append(adjusted_line)

        # 새로운 파일에 저장
        with open(output_path, 'w') as output_file:
            output_file.write('\n'.join(output_lines))


def xywh_to_xyxy(
        x: float,
        y: float,
        w: float,
        h: float
) -> Tuple[float, float, float, float]:
    
    x_center = x
    y_center = y
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return x1, y1, x2, y2


def xyxy_to_xywh(
        x1: float,
        y1: float,
        x2: float,
        y2: float
)-> Tuple[float, float, float, float]:
    
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return x_center, y_center, w, h
