import os
from PIL import Image
from datetime import datetime
from utils.file_operations import delete_files_with_classes, moving_images
from utils.bbox_operations import denormalize_xyxy, xywh_to_xyxy, process_yolo_label
from utils.text_operations import sanitize_filename, replace_first_column_to_word, replace_first_column_to_0

def process_image(
        image_file_path: str,
        text_file_path: str,
        output_folder_path: str,
        output_text_file: str,
        idx_1: int
) -> None:
    
    # 텍스트 파일 읽기
    with open(text_file_path, 'r', encoding='cp949') as text_file:
        lines = text_file.readlines()

    # 이미지 읽기
    image = Image.open(image_file_path)
    image_width, image_height = image.size

    with open(output_text_file, 'a') as text_file:

        for idx_2, line in enumerate(lines):
            values = line.strip().split()
            word = values[0]
            win_word = sanitize_filename(word)
            x, y, w, h = map(float, values[1:])
            xyxy_coords = xywh_to_xyxy(x, y, w, h)

            # 이미지 자르기
            xyxy_coords = denormalize_xyxy(xyxy_coords, image_width, image_height)
            cropped_image = image.crop(xyxy_coords)

            # 현재 날짜 및 시간으로 파일 이름 생성
            current_datetime = datetime.now().strftime("%Y%m%d")  # ("%Y%m%d%H%M%S")
            new_image_name = f"{win_word}_{idx_1}_{idx_2}_{current_datetime}.jpg"  # 이미지 파일 확장자는 사용하는 형식에 맞게 수정
            new_image_path = f'{output_folder_path}/{new_image_name}'

            # 이미지 저장
            cropped_image.save(new_image_path)
            print(f"Processed and saved: {new_image_path}                   ", end='\r', flush=True)

            # 텍스트 파일에 경로와 단어 기록
            new_image_path = new_image_path.split('/')[-3:]
            new_image_path = '/'.join(new_image_path)
            text_file.write(f"./{new_image_path} {word}\n")


def process_text(folder_path: str) -> None:

    for filename1 in os.listdir(folder_path):
        if filename1.endswith('.txt') and 'classes' not in filename1:
            # 파일 이름에서 숫자 추출
            number = filename1.split('_')[-1].split('.')[0]
            
            # 해당 숫자에 대응하는 두 번째 파일 찾기
            filename2 = f'classes_{number}.txt'
            
            # 결과 파일 이름 지정
            output_filename = filename1
            
            # 함수 호출
            replace_first_column_to_word(
                os.path.join(folder_path, filename1),
                os.path.join(folder_path, filename2), 
                os.path.join(folder_path, output_filename)
            )
            

def preprocessing(target_path: str, mode: str) -> None:

    TARGET_PATH = target_path
    MODE = mode

    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)

    # 특정 폴더에 있는 모든 텍스트 파일 처리
    process_text(TARGET_PATH)

    # 특정 폴더에서 'classes'가 들어간 파일 삭제
    delete_files_with_classes(TARGET_PATH)

    # YOLO 라벨 처리
    process_yolo_label(TARGET_PATH, TARGET_PATH)

    # 특정 폴더에 있는 모든 텍스트 파일 처리
    for idx_1, image_file in enumerate(os.listdir(TARGET_PATH)):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # 파일 이름에서 숫자 추출
            number = image_file.split('.')[0]
            
            # 해당 숫자에 대응하는 두 번째 파일 찾기
            txt_file = f'{number}.txt'
            
            output_folder_path = f'datasets/ocr_data/images/{MODE}'
            output_text_file = f'datasets/ocr_data/annotation_{MODE}.txt'
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # 함수 호출
            process_image(
                os.path.join(TARGET_PATH, image_file),
                os.path.join(TARGET_PATH, txt_file),
                output_folder_path,
                output_text_file,
                idx_1
            )

    output_folder_path = f'datasets/yolo_data/{MODE}/labels'
    replace_first_column_to_0(TARGET_PATH, output_folder_path)

    output_folder_path = f'datasets/yolo_data/{MODE}/images'
    moving_images(TARGET_PATH, output_folder_path)
