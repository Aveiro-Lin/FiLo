import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp_visdrone/weights/best.pt')
    
    # Key: Add these parameters to generate the confusion matrix
    model.val(
        data='ultralytics/cfg/datasets/VisDrone.yaml',
        split='val',
        imgsz=640,
        batch=16,
        iou=0.5,
        conf=0.001,
        workers=0,
        device='0',
        save_txt=True,
        save_conf=True,
        project='runs/val',
        name='Visdrone',
        # Add the following parameters ↓↓↓
        save_json=True,      # Save the prediction results in JSON format
        save_hybrid=True,    # Save the mixed format (for obfuscation matrices)
        plots=True,          # Generate various charts, including confusion matrices
        verbose=True,        # Display detailed information
        exist_ok=True        # Overwrite the existing results
    )
