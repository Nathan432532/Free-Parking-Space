import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import json
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('models/parking_model.pth', map_location=device))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_slot_crop(img, pts):
    pts_src = np.array(pts, dtype=np.float32)
    pts_dst = np.array([[0, 0], [64, 0], [64, 64], [0, 64]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img, matrix, (64, 64))

def detect_parking(img_path, json_path):
    img = cv2.imread(img_path)

    with open(json_path, 'r') as f:
        slots = json.load(f)

    with torch.no_grad():
        for pts in slots:
            crop = get_slot_crop(img, pts)
            input_tensor = preprocess(crop).unsqueeze(0).to(device)

            outputs = model(input_tensor)
            conf, preds = torch.max(outputs, 1)
            confidence = conf.item() * 100
            is_occupied = preds.item() == 1

            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            label = f"{'Bezet' if is_occupied else 'Vrij'} ({confidence:.0f}%)"
            cv2.polylines(img, [np.array(pts)], True, color, 2)
            cv2.putText(img, label, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Parking Slot Detection', img)
    print("Detectie voltooid. Druk op een toets om te sluiten.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_parking('data/raw/nearly_empty/test.jpg', 'config/parking_slots.json')