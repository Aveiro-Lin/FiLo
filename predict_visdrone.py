from ultralytics import YOLO

# Load Model
model = YOLO(r"runs/train/exp_visdrone/weights/best.pt")

# Infer and Save Results
results = model.predict(
    source=r"/opt/liblibai-models/user-workspace2/users/syq/Project/RSDATA/VISDRONE/images/test",
    save=True,          # Save Detection Visualization Graph
    save_txt=True,      # Save txt file
    save_conf=True,     # Save Confidence Level
    project=r"runs/detect",
    name="expvisdrone"
)
