import torch
import argparse
import evaluate
import numpy as np
import pandas as pd
from pynvml import *
from PIL import Image
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from transformers import default_data_collator
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.optim import SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSprop
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

def print_gpu_utilization() -> None:

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    

def preprocess_data(file_path: str) -> pd.DataFrame:

    # 파일 읽기
    df = pd.read_csv(file_path, header=None, sep=" ", encoding='cp949')
    
    # 열 이름 변경
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    
    # 'text' 열의 공백 제거
    df['text'] = df['text'].str.strip()
    
    # 'text' 값이 누락된 행 제거
    df = df.dropna(subset=['text'])
    
    # 인덱스 재설정
    df.reset_index(drop=True, inplace=True)
    
    return df


def initialize_ocr_model(
        trocr_model: str
) -> Tuple[VisionEncoderDecoderModel, AutoTokenizer, TrOCRProcessor]:
    
    # OCR 모델 초기화
    model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
    
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(trocr_model)
    
    # OCR Processor 초기화
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    return model, tokenizer, processor


class OCRDataset(Dataset):
    def __init__(self, dataset_dir: str, df: pd.DataFrame, processor: TrOCRProcessor,
                 tokenizer: AutoTokenizer, max_target_length: int = 32):
        self.dataset_dir = dataset_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[int]]]:

        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # prepare image (i.e. resize + normalize)
        image = Image.open(self.dataset_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text      
        labels = self.tokenizer(text, padding="max_length", 
                                stride=32,
                                truncation=True,
                                max_length=self.max_target_length).input_ids
        
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]
        
        # Construct the encoding dictionary
        encoding: Dict[str, Union[torch.Tensor, List[int]]] = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    

def create_datasets(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_dir: str,
        tokenizer: AutoTokenizer,
        processor: TrOCRProcessor,
        max_length: int
) -> Tuple[Dataset, Dataset]:

    train_dataset = OCRDataset(
        dataset_dir=dataset_dir,
        df=train_df,
        tokenizer=tokenizer,
        processor=processor,
        max_target_length=max_length
    )

    eval_dataset = OCRDataset(
        dataset_dir=dataset_dir,
        df=test_df,
        tokenizer=tokenizer,
        processor=processor,
        max_target_length=max_length
    )

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    return train_dataset, eval_dataset


def configure_model(
        model: VisionEncoderDecoderModel,
        tokenizer: AutoTokenizer,
        max_length: int
) -> VisionEncoderDecoderModel:
    
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model


def compute_metrics(pred: Any) -> Dict[str, float]:

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}


def ocr_train(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        num_train_epochs: int,
        batch_size: int,
        learning_rate: float,
        save_splits: int,
        eval_splits: int,
        resume: bool
) -> None:
    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        fp16=True,
        learning_rate=learning_rate,
        output_dir="./",
        logging_dir="./logs",
        logging_steps=100,
        save_steps=np.ceil(len(train_dataset)/(save_splits*batch_size)),
        eval_steps=np.ceil(len(train_dataset)/(eval_splits*batch_size)),
    )

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.9999)

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )

    print_gpu_utilization()
    
    checkpoint_folder = '.'
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_folder)

    if latest_checkpoint_path and resume:
        trainer.train(latest_checkpoint_path)
    else:
        trainer.train()
    trainer.save_model(output_dir="./model")


def find_latest_checkpoint(folder_path: str) -> Optional[str]:
    checkpoints = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('checkpoint-')]

    if not checkpoints:
        return None

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(folder_path, latest_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TrOCR Training Script')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=10e-6, help='Initial learning rate')
    parser.add_argument('--save', type=int, default=8, help='Save splits')
    parser.add_argument('--eval', type=int, default=8, help='Eval splits')
    parser.add_argument('--resume', action='store_true', help='Resume training')

    args = parser.parse_args()

    # 훈련 데이터 전처리
    train_df = preprocess_data('datasets/ocr_data/annotation_train.txt')

    # 테스트 데이터 전처리
    test_df = preprocess_data('datasets/ocr_data/annotation_val.txt')

    # OCR 모델 및 토크나이저 초기화
    model, tokenizer, processor = initialize_ocr_model('team-lucid/trocr-small-korean')

    dataset_dir: str = 'datasets/ocr_data'
    max_length: int = 64

    train_dataset, eval_dataset = create_datasets(
        train_df,
        test_df,
        dataset_dir,
        tokenizer,
        processor,
        max_length
    )
    model = configure_model(model, tokenizer, max_length)
    ocr_train(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        num_train_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        save_splits=args.save,
        eval_splits=args.eval,
        resume=args.resume
    )
