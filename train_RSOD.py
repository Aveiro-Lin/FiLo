import os

os.environ['WANDB_DISABLED'] = 'true'
# from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics/cfg/models/new/FiLo_v0s.yaml')
    model.train(data=r'ultralytics/cfg/datasets/RSOD.yaml',
                imgsz=640,
                epochs=600,
                batch=16,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp_RSOD',
                single_cls=False,
                cache=False,
                )

#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)

