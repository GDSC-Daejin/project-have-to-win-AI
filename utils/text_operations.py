# text_operations.py
import re
import os

def replace_first_column_to_word(
        input_file1: str,
        input_file2: str,
        output_file: str
) -> None:
    
    # 파일 1 읽기
    with open(input_file1, 'r', encoding='cp949') as file1:
        lines_file1 = file1.readlines()

    # 파일 2 읽기
    with open(input_file2, 'r', encoding='utf-8') as file2:
        lines_file2 = file2.readlines()

    # 결과를 저장할 파일 열기
    with open(output_file, 'w') as output:
        # 파일 1의 각 줄에 대해 처리
        for line1 in lines_file1:
            # 각 줄을 공백을 기준으로 나누어 리스트로 변환
            values = line1.split()
            # 파일 2에서 해당 인덱스의 문자열 가져오기
            index = int(values[0])  # 첫 번째 열의 번호
            if 0 <= index < len(lines_file2):
                replaced_line = f"{lines_file2[index].strip()} {' '.join(values[1:])}"
                # 결과 파일에 쓰기
                output.write(replaced_line + '\n')


def replace_first_column_to_0(input_folder_path, output_folder_path):

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    file_list = [f for f in os.listdir(input_folder_path) if f.endswith('.txt')]

    for file_name in file_list:
        file_path = os.path.join(input_folder_path, file_name)
        output_path = os.path.join(output_folder_path, file_name)

        with open(file_path, 'r', encoding='cp949') as file:
            lines = file.readlines()

        # 각 라인의 첫 번째 값을 0으로 바꾸기
        modified_lines = ['0 ' + line.split(' ', 1)[1] for line in lines]

        with open(output_path, 'w') as file:
            file.writelines(modified_lines)


def sanitize_filename(filename):
    
    # 윈도우즈 파일명으로 사용할 수 없는 문자 패턴 정의
    invalid_chars = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

    # 유효하지 않은 문자를 빈 문자열로 대체
    sanitized_filename = re.sub(invalid_chars, '', filename)

    return sanitized_filename