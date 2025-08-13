from data_pipeline import create_yolo_subset
from train_yolo import train_yolo

VOC_ROOT = r"C:\IDD_Detection"
OUTPUT_ROOT = r"C:\Users\Asus\Downloads\yolo_project\mini_idd_yolo"

if __name__ == "__main__":
    print("Preparing dataset...")
    yaml_path = create_yolo_subset(VOC_ROOT, OUTPUT_ROOT, num_train=250, num_val=50)
    print(f"Dataset ready: {yaml_path}")

    print("Starting YOLOv5 training...")
    train_yolo(data_yaml=yaml_path, weights="yolov5s.pt", img_size=416, batch_size=1, epochs=10)
