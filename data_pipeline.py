import os
import random
import xml.etree.ElementTree as ET
import shutil

CLASSES = ["car", "bus", "truck", "person", "motorbike", "bicycle"]

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

def create_yolo_subset(VOC_ROOT, OUTPUT_ROOT, num_train=250, num_val=50):
    images_dir = os.path.join(VOC_ROOT, "JPEGImages")
    annotations_dir = os.path.join(VOC_ROOT, "Annotations")
    all_ids = [f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.shuffle(all_ids)
    
    train_ids = all_ids[:num_train]
    val_ids = all_ids[num_train:num_train+num_val]

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in ids:
            img_src = os.path.join(images_dir, img_id + ".jpg")
            xml_src = os.path.join(annotations_dir, img_id + ".xml")
            img_dst = os.path.join(OUTPUT_ROOT, "images", split, img_id + ".jpg")
            label_dst = os.path.join(OUTPUT_ROOT, "labels", split, img_id + ".txt")

            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            os.makedirs(os.path.dirname(label_dst), exist_ok=True)

            shutil.copy(img_src, img_dst)
            convert_annotation(xml_src, label_dst)

    # Save YAML
    yaml_content = f"""
train: {os.path.join(OUTPUT_ROOT, 'images/train').replace(os.sep, '/')}
val: {os.path.join(OUTPUT_ROOT, 'images/val').replace(os.sep, '/')}
nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = os.path.join(OUTPUT_ROOT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    return yaml_path
