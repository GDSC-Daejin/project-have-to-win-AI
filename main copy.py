import os
import shutil
import zipfile
import threading
from datetime import datetime
from subprocess import Popen, PIPE, STDOUT
from typing import Iterator, List, NoReturn
from modules.preprocessing import preprocessing
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

UPLOAD_FOLDER: str = 'uploads'
ALLOWED_EXTENSIONS: set[str] = {'zip'}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 추가한 부분
#from fastapi.middleware.cors import CORSMiddleware
#
# CORS 설정
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # 실제 사용시 특정 origin을 지정해주는 것이 보안상 좋습니다.
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)



# 서브프로세스 및 로그 관련 변수
yolo_process: Popen[str] = None
trocr_process: Popen[str] = None
log_buffer: List[str] = []
log_entry_list: List[str] = []


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 파일 업로드 함수
def save_uploaded_files(directory: str, files: List[UploadFile]) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

    for file in files:
        if file and allowed_file(file.filename):
            file_path = os.path.join(directory, file.filename)
            with open(file_path, 'wb') as f:
                f.write(file.file.read())
            print(f'파일 {file.filename} 업로드 성공!')
            
            # 업로드된 파일이 zip 파일인지 확인
            if file.filename.endswith('.zip'):
                # zip 파일의 내용을 압축 해제
                zip_file_path = file_path
                extract_path = directory
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f'파일 {file.filename} 압축 풀기 완료!')
                
                # 원본 zip 파일 삭제
                os.remove(zip_file_path)
                print(f'파일 {file.filename} 삭제 완료!')

        else:
            print(f'파일 {file.filename} 업로드 실패 - 허용되지 않는 확장자')


# YOLO Training 서브프로세스 실행 함수
def start_yolo_train(
        model: str,
        data: str,
        epochs: int,
        patience: int,
        batch: int,
        imgsz: int,
        project: str,
        name: str,
        pretrained: bool,
        optimizer: str,
        seed: int,
        cos_lr: bool,
        resume: bool,
        lr0: float,
        lrf: float
) -> None:
        
    global yolo_process, log_buffer
    global log_entry_list
    log_buffer.clear()

    # Construct the command based on the received arguments
    command: List[str] = [
        'python', 'modules/yolo_train.py',
        '--model', model,
        '--data', data,
        '--epochs', str(epochs),
        '--patience', str(patience),
        '--batch', str(batch),
        '--imgsz', str(imgsz),
        '--project', project,
        '--name', name,
        '--pretrained' if pretrained else '',
        '--optimizer', optimizer,
        '--seed', str(seed),
        '--cos_lr' if cos_lr else '',
        '--resume' if resume else '',
        '--lr0', str(lr0),
        '--lrf', str(lrf)
    ]

    command_str: str = ' '.join(command)

    yolo_process = Popen(
        command_str,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1,
        universal_newlines=True
)

    while yolo_process.poll() is None:
        line: str = yolo_process.stdout.readline()
        log_buffer.append(line)
    
    create_log_file(log_entry_list, 'YOLO')


# TrOCR Training 서브프로세스 실행 함수
def start_trocr_train(
        epochs: int,
        batch: int,
        lr: float,
        save_splits: int,
        eval_splits: int,
        resume: bool
) -> None:
    
    global trocr_process, log_buffer
    global log_entry_list
    log_buffer.clear()

    # Construct the command based on the received arguments
    command: List[str] = [
        'python', 'modules/trocr_train.py',
        '--epochs', str(epochs),
        '--batch', str(batch),
        '--lr', str(lr),
        '--save', str(save_splits),
        '--eval', str(eval_splits),
        '--resume' if resume else ''
    ]

    command_str: str = ' '.join(command)

    trocr_process = Popen(
        command_str,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1,
        universal_newlines=True
)

    while trocr_process.poll() is None:
        line: str = trocr_process.stdout.readline()
        log_buffer.append(line)
    
    create_log_file(log_entry_list, 'TrOCR')


# YOLO Training 서브프로세스 중단 함수
def stop_yolo_train() -> NoReturn:
    global yolo_process
    global log_entry_list
    if yolo_process is not None and yolo_process.poll() is None:
        yolo_process.terminate()
        create_log_file(log_entry_list, 'YOLO')


# TrOCR Training 서브프로세스 중단 함수
def stop_trocr_train() -> NoReturn:
    global trocr_process
    global log_entry_list
    if trocr_process is not None and trocr_process.poll() is None:
        trocr_process.terminate()
        create_log_file(log_entry_list, 'TrOCR')


# 주기적으로 로그를 확인하여 브라우저에 업데이트
def check_logs() -> Iterator[str]:
    global log_buffer
    global log_entry_list
    log_entry_list.clear()
    while True:
        if log_buffer:
            log_entry = log_buffer.pop(0)
            log_entry_list.append(log_entry)

            yield f'data: {log_entry}\n\n'.encode('utf-8')


def create_log_file(log_entry_list: List[str], model: str) -> None:
    log_folder = 'log'

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_folder, f"{model}_log_file_{current_time}.txt")

    with open(log_file_path, 'w', encoding='utf-8') as file:
        file.writelines(log_entry_list)




@app.get('/')
async def index(request: Request):
    return {"message": "Welcome to the index page!"}


@app.get('/upload')
async def upload(request: Request):
    return {"message": "Upload page"}


#@app.post('/upload')
#async def upload(request: Request):
#    form_data = await request.form()
#    img_files = form_data.getlist('file')
#    if img_files:
#        save_uploaded_files(UPLOAD_FOLDER, img_files)
#    return {"message": "Files uploaded successfully"}



@app.post('/upload')
async def upload(request: Request):
    try:
        form_data = await request.form()
        img_files = form_data.getlist('file')
        if img_files:
            save_uploaded_files(UPLOAD_FOLDER, img_files)
            return {"message": "Files uploaded successfully"}
        else:
            raise HTTPException(status_code=400, detail="No files were provided in the request.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file upload: {str(e)}")



@app.get('/upload2')  # 엔드포인트 이름 변경
async def upload2(request: Request):  # 함수 이름 변경
    return {"message": "Upload2 page"}


@app.post('/upload2')  # 엔드포인트 이름 변경
async def upload2(request: Request):  # 함수 이름 변경
    form_data = await request.form()
    img_files = form_data.getlist('file')
    if img_files:
        save_uploaded_files(UPLOAD_FOLDER, img_files)
    return {"message": "Files uploaded successfully"}


@app.post('/delete_uploads')
async def delete_uploads(request: Request):
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            print(f"폴더 '{UPLOAD_FOLDER}' 삭제 성공!")
        else:
            print(f"폴더 '{UPLOAD_FOLDER}' 삭제 실패 - 존재하지 않는 폴더")
    except Exception as e:
        print(f"폴더 '{UPLOAD_FOLDER}' 삭제 중 오류 발생: {str(e)}")

    return {"message": "Uploads deleted successfully"}


@app.get('/preprocess')
async def preprocess(request: Request):
    return {"message": "Preprocess page"}


@app.post('/preprocess')
async def preprocess(request: Request):
    form_data = await request.form()
    mode = form_data.get('mode')
    preprocessing(UPLOAD_FOLDER, mode)
    return {"message": "Preprocessing completed"}


@app.post('/delete_datasets')
async def delete_datasets(request: Request):
    target = 'datasets'
    try:
        if os.path.exists(target):
            shutil.rmtree(target)
            print(f"폴더 '{target}' 삭제 성공!")
        else:
            print(f"폴더 '{target}' 삭제 실패 - 존재하지 않는 폴더")
    except Exception as e:
        print(f"폴더 '{target}' 삭제 중 오류 발생: {str(e)}")

    return {"message": "Datasets deleted successfully"}


@app.get('/yolo_train')
async def train_yolo(request: Request):
    return {"message": "YOLO training page"}


@app.post('/yolo_train')
async def train_yolo(
        epochs: int,
        batch: int,
        lr0: float,
        resume: bool
):
    model = 'yolov8n.pt'
    data = 'cfg/datasets/text.yaml'
    patience = 0
    imgsz = 640
    project = 'yolo_runs'
    name = 'yolo_train'
    optimizer = 'SGD'
    seed = 0
    lrf = 0.01
    pretrained = True
    cos_lr = False

    threading.Thread(
        target=start_yolo_train,
        args=(
            model,
            data,
            epochs,
            patience,
            batch,
            imgsz,
            project,
            name,
            pretrained,
            optimizer,
            seed,
            cos_lr,
            resume,
            lr0,
            lrf
        )
    ).start()
    return {"message": "YOLO training started"}



@app.get('/trocr_train')
async def train_trocr(request: Request):
    return {"message": "TrOCR training page"}


@app.post('/trocr_train')
async def train_trocr(
        request: Request,
        epochs: int,
        batch: int,
        lr: float,
        save_splits: int,
        eval_splits: int,
        resume: bool
):

    threading.Thread(
        target=start_trocr_train,
        args=(
            epochs,
            batch,
            lr,
            save_splits,
            eval_splits,
            resume
        )
    ).start()

    return {"message": "TrOCR training started"}


@app.post('/stop_yolo_train')
async def stop_yolo(request: Request):
    threading.Thread(target=stop_yolo_train).start()
    return {"message": "Stopping YOLO training"}


@app.post('/stop_trocr_train')
async def stop_trocr(request: Request):
    threading.Thread(target=stop_trocr_train).start()
    return {"message": "Stopping TrOCR training"}


@app.get('/stream_logs')
async def stream_logs(request: Request):
    return StreamingResponse(check_logs(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.219.161", port=8080)