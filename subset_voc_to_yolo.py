import os
import random
import shutil
import xml.etree.ElementTree as ET

# ==== CONFIG ====
VOC_ROOT = r"C:\IDD_Detection"
OUTPUT_ROOT = r"C:\Users\Asus\Downloads\ml-app-cicd-main\mini_idd_yolo" 
NUM_TRAIN = 250
NUM_VAL = 50
CLASSES = ["car", "bus", "truck", "person", "motorbike", "bicycle"]

# Create output folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

def voc_to_yolo_bbox(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

def convert_annotation(xml_path, yolo_txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    with open(yolo_txt_path, "w") as f:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in CLASSES:
                continue
            cls_id = CLASSES.index(cls)
            xml_box = obj.find("bndbox")
            b = (
                int(xml_box.find("xmin").text),
                int(xml_box.find("ymin").text),
                int(xml_box.find("xmax").text),
                int(xml_box.find("ymax").text)
            )
            bb = voc_to_yolo_bbox(b, img_w, img_h)
            f.write(f"{cls_id} {' '.join(map(str, bb))}\n")

def read_id_list(file_path):
    with open(file_path, "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]

train_ids = read_id_list(os.path.join(VOC_ROOT, "train.txt"))
val_ids = read_id_list(os.path.join(VOC_ROOT, "val.txt"))

train_ids = random.sample(train_ids, NUM_TRAIN)
val_ids = random.sample(val_ids, NUM_VAL)

for img_id in train_ids:
    img_src = os.path.join(VOC_ROOT, "JPEGImages", img_id + ".jpg")
    xml_src = os.path.join(VOC_ROOT, "Annotations", img_id + ".xml")
    img_dst = os.path.join(OUTPUT_ROOT, "images", "train", img_id + ".jpg")
    label_dst = os.path.join(OUTPUT_ROOT, "labels", "train", img_id + ".txt")
    
    # Make sure parent directories exist
    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    os.makedirs(os.path.dirname(label_dst), exist_ok=True)
    
    shutil.copy(img_src, img_dst)
    convert_annotation(xml_src, label_dst)

for img_id in val_ids:
    img_src = os.path.join(VOC_ROOT, "JPEGImages", img_id + ".jpg")
    xml_src = os.path.join(VOC_ROOT, "Annotations", img_id + ".xml")
    img_dst = os.path.join(OUTPUT_ROOT, "images", "val", img_id + ".jpg")
    label_dst = os.path.join(OUTPUT_ROOT, "labels", "val", img_id + ".txt")
    
    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    os.makedirs(os.path.dirname(label_dst), exist_ok=True)
    
    shutil.copy(img_src, img_dst)
    convert_annotation(xml_src, label_dst)


yaml_content = f"""
train: {OUTPUT_ROOT.replace(os.sep, '/')}/images/train
val: {OUTPUT_ROOT.replace(os.sep, '/')}/images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
with open(os.path.join(OUTPUT_ROOT, "data.yaml"), "w") as f:
    f.write(yaml_content)

print(f"✅ Subset created at: {OUTPUT_ROOT}")
print(f"Training images: {len(train_ids)}, Validation images: {len(val_ids)}")
