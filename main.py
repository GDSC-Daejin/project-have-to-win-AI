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

UPLOAD_FOLDER: str = 'uploads'
ALLOWED_EXTENSIONS: set[str] = {'zip'}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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


# YOLO Training 서브프로세스 중단 함수
def stop_yolo_train() -> NoReturn:
    global yolo_process
    global log_entry_list
    model = 'YOLO'
    if yolo_process is not None and yolo_process.poll() is None:
        yolo_process.terminate()
        create_log_file(log_entry_list, model)


# TrOCR Training 서브프로세스 중단 함수
def stop_trocr_train() -> NoReturn:
    global trocr_process
    global log_entry_list
    model = 'TrOCR'
    if trocr_process is not None and trocr_process.poll() is None:
        trocr_process.terminate()
        create_log_file(log_entry_list, model)


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


@app.route('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.route('/upload', methods=['GET', 'POST'])
async def upload(request: Request):
    if request.method == 'POST':
        form_data = await request.form()
        img_files = form_data.getlist('file')
        if img_files:
            save_uploaded_files(UPLOAD_FOLDER, img_files)
    return templates.TemplateResponse("upload.html", {"request": request})


@app.route('/delete_folder', methods=['POST'])
async def delete_folder(request: Request):
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            print(f"폴더 '{UPLOAD_FOLDER}' 삭제 성공!")
        else:
            print(f"폴더 '{UPLOAD_FOLDER}' 삭제 실패 - 존재하지 않는 폴더")
    except Exception as e:
        print(f"폴더 '{UPLOAD_FOLDER}' 삭제 중 오류 발생: {str(e)}")

    return RedirectResponse(url=request.url_for("upload"), status_code=303)


@app.route('/preprocess', methods=['GET', 'POST'])
async def preprocess(request: Request):
    if request.method == 'POST':
        form_data = await request.form()
        mode = form_data.get('mode')
        preprocessing(UPLOAD_FOLDER, mode)
    return templates.TemplateResponse("preprocess.html", {"request": request})


@app.route('/yolo_train', methods=['GET', 'POST'])
async def train_yolo(request: Request):
    if request.method == 'POST':
        form_data = await request.form()
        model = form_data.get('model')
        data = form_data.get('data')
        epochs = int(form_data.get('epochs'))
        patience = int(form_data.get('patience'))
        batch = int(form_data.get('batch'))
        imgsz = int(form_data.get('imgsz'))
        project = form_data.get('project')
        name = form_data.get('name')
        pretrained = 'pretrained' in form_data
        optimizer = form_data.get('optimizer')
        seed = int(form_data.get('seed'))
        cos_lr = 'cos_lr' in form_data
        resume = 'resume' in form_data
        lr0 = float(form_data.get('lr0'))
        lrf = float(form_data.get('lrf'))

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

    return templates.TemplateResponse("yolo_train.html", {"request": request})


@app.route('/trocr_train', methods=['GET', 'POST'])
async def train_trocr(request: Request):
    if request.method == 'POST':
        form_data = await request.form()
        epochs = int(form_data.get('epochs'))
        batch = int(form_data.get('batch'))
        lr = float(form_data.get('lr'))
        save_splits = int(form_data.get('save'))
        eval_splits = int(form_data.get('eval'))
        resume = 'resume' in form_data

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

    return templates.TemplateResponse("trocr_train.html", {"request": request})


@app.route('/stop_yolo_train', methods=['POST'])
async def stop_yolo(request: Request):
    threading.Thread(target=stop_yolo_train).start()
    return templates.TemplateResponse("yolo_train.html", {"request": request})


@app.route('/stop_trocr_train', methods=['POST'])
async def stop_trocr(request: Request):
    threading.Thread(target=stop_trocr_train).start()
    return templates.TemplateResponse("trocr_train.html", {"request": request})


@app.route('/stream_logs', methods=['GET'])
async def stream_logs(request: Request):
    return StreamingResponse(check_logs(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)