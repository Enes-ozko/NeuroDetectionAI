import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class RemoveSkullAndFace(object):
   
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img

        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        img_cleaned = cv2.bitwise_and(img_np, img_np, mask=mask)

        return Image.fromarray(img_cleaned)

def get_transforms(mode: str, image_size: int = 224):
 
    if mode == "train":
        return transforms.Compose([
            RemoveSkullAndFace(), 
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    return transforms.Compose([
        RemoveSkullAndFace(), 
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])