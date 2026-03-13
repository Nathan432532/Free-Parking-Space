import json
import cv2
import os

def convert_to_coco_format(base_path, set_name, num_img):
    json_path = os.path.join(base_path, set_name, '_annotations.coco.json')
    img_dir = os.path.join(base_path, set_name)
    output_dir = f"data/processed/{set_name}"

    with open(json_path) as f:
        data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    for cat_name in categories.values():
        os.makedirs(os.path.join(output_dir, cat_name), exist_ok=True)
    
    for i, ann in enumerate(data['annotations']):
        if i >= num_img: break
        img_info = next(img for img in data['images'] if img['id'] == ann['image_id'])
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        img = cv2.imread(img_path)
        if img is None: continue

        x, y, w, h = map(int, ann['bbox'])
        crop = img[max(0, y): y+h, max(0, x): x+w]

        if crop.size > 0:
            label = categories[ann['category_id']]
            save_path = os.path.join(output_dir, label, f"{ann['id']}.jpg")
            cv2.imwrite(save_path, crop)

convert_to_coco_format('data/raw', 'train', 10000)
convert_to_coco_format('data/raw', 'test', 1000)
convert_to_coco_format('data/raw', 'valid', 200)
