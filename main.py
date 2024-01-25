import os
import threading
from subprocess import Popen, PIPE, STDOUT
from typing import Iterator, List, NoReturn
from modules.preprocessing import preprocessing
from flask import Flask, render_template, request

UPLOAD_FOLDER: str = 'uploads'
ALLOWED_EXTENSIONS: set[str] = {'txt', 'png', 'jpg', 'jpeg'}

app: Flask = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 서브프로세스 및 로그 관련 변수
yolo_process: Popen[str] = None
trocr_process: Popen[str] = None
log_buffer: list = []


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 파일 업로드 함수
def save_uploaded_files(directory: str, files: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

    for file in files:
        if file and allowed_file(file.filename):
            with open(os.path.join(directory, file.filename), 'wb') as f:
                f.write(file.read())
            print(f'파일 {file.filename} 업로드 성공!')
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
    if yolo_process is not None and yolo_process.poll() is None:
        yolo_process.terminate()


# TrOCR Training 서브프로세스 중단 함수
def stop_trocr_train() -> NoReturn:
    global trocr_process
    if trocr_process is not None and trocr_process.poll() is None:
        trocr_process.terminate()


# 주기적으로 로그를 확인하여 브라우저에 업데이트
def check_logs() -> Iterator[str]:
    global log_buffer
    while True:
        if log_buffer:
            yield 'data: ' + log_buffer.pop(0).replace('\n', '<br>') + '\n\n'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img_files = request.files.getlist('file')
        if img_files:
            save_uploaded_files(app.config['UPLOAD_FOLDER'], img_files)
    return render_template('upload.html')


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if request.method == 'POST':
        mode = request.form.get('mode')
        preprocessing(app.config['UPLOAD_FOLDER'], mode)
    return render_template('preprocess.html')


@app.route('/yolo_train', methods=['GET', 'POST'])
def train_yolo():
    if request.method == 'POST':
        model = request.form.get('model')
        data = request.form.get('data')
        epochs = int(request.form.get('epochs'))
        patience = int(request.form.get('patience'))
        batch = int(request.form.get('batch'))
        imgsz = int(request.form.get('imgsz'))
        project = request.form.get('project')
        name = request.form.get('name')
        pretrained = 'pretrained' in request.form
        optimizer = request.form.get('optimizer')
        seed = int(request.form.get('seed'))
        cos_lr = 'cos_lr' in request.form
        resume = 'resume' in request.form
        lr0 = float(request.form.get('lr0'))
        lrf = float(request.form.get('lrf'))

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

    return render_template('yolo_train.html')


@app.route('/trocr_train', methods=['GET', 'POST'])
def train_trocr():
    if request.method == 'POST':
        epochs = int(request.form.get('epochs'))
        batch = int(request.form.get('batch'))
        lr = float(request.form.get('lr'))
        save_splits = int(request.form.get('save'))
        eval_splits = int(request.form.get('eval'))
        resume = 'resume' in request.form

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

    return render_template('trocr_train.html')


@app.route('/stop_yolo_train', methods=['POST'])
def stop_yolo():
    threading.Thread(target=stop_yolo_train).start()
    return render_template('yolo_train.html')


@app.route('/stop_trocr_train', methods=['POST'])
def stop_trocr():
    threading.Thread(target=stop_trocr_train).start()
    return render_template('trocr_train.html')


@app.route('/stream_logs')
def stream_logs():
    return check_logs(), 200, {'Content-Type': 'text/event-stream'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
