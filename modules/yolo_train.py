import argparse
from ultralytics import YOLO

def yolo_train(
        model: str = 'yolov8n.pt',
        data: str = 'cfg/datasets/text.yaml',
        epochs: int = 100,
        patience: int = 0,
        batch: int = 16,
        imgsz: int = 640,
        project: str = 'runs',
        name: str = 'train',
        pretrained: bool = True,
        optimizer: str = 'auto',
        seed: int = 0,
        cos_lr: bool = False,
        resume: bool = False,
        lr0: float = 0.01,
        lrf: float = 0.01
) -> None:
    
    model = YOLO(model)

    # Train the model
    if resume:
        model.train(resume=resume)
    else:
        model.train(
            data=data,
            epochs=epochs,
            patience=patience,
            batch=batch,
            imgsz=imgsz,
            project=project,
            name=name,
            pretrained=pretrained,
            optimizer=optimizer,
            seed=seed,
            cos_lr=cos_lr,
            resume=resume,
            lr0=lr0,
            lrf=lrf
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    parser.add_argument('--model', type=str, help='Path to the model file (without extension)')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=0, help='Patience for early stopping')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, help='Project name')
    parser.add_argument('--name', type=str, help='Run name')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer type')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cos_lr', action='store_true', help='Use cosine learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate')

    args = parser.parse_args()

    yolo_train(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        seed=args.seed,
        cos_lr=args.cos_lr,
        resume=args.resume,
        lr0=args.lr0,
        lrf=args.lrf
    )
