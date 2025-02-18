import hydra
import torch
import os
from pathlib import Path

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import easyocr
import cv2

# Create necessary directories
IMAGES_DIR = Path('images')  # Directory for input images
OUTPUT_DIR = Path('runs/detect')  # Directory for output
IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize EasyOCR with error handling
def initialize_reader():
    try:
        return easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        print("Attempting to initialize without GPU...")
        return easyocr.Reader(['en'], gpu=False)

reader = initialize_reader()

def perform_ocr_on_image(img, coordinates):
    try:
        x, y, w, h = map(int, coordinates)
        # Ensure coordinates are within image bounds
        y = max(0, y)
        h = min(h, img.shape[0])
        x = max(0, x)
        w = min(w, img.shape[1])
        
        cropped_img = img[y:h, x:w]
        
        # Check if cropped image is empty
        if cropped_img.size == 0:
            return ""
            
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        results = reader.readtext(gray_img)
        
        text = ""
        for res in results:
            if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
                text = res[1]
                
        return str(text)
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return ""

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                      self.args.conf,
                                      self.args.iou,
                                      agnostic=self.args.agnostic_nms,
                                      max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(OUTPUT_DIR / p.name)
        self.txt_path = str(OUTPUT_DIR / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
            
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
            
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                text_ocr = perform_ocr_on_image(im0, xyxy)
                label = text_ocr if text_ocr else f'{self.model.names[c]} {conf:.2f}'
                self.annotator.box_label(xyxy, label, color=colors(c, True))

            if self.args.save_crop:
                save_one_box(xyxy,
                           im0.copy(),
                           file=OUTPUT_DIR / 'crops' / self.model.names[c] / f'{p.stem}.jpg',
                           BGR=True)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = str(IMAGES_DIR)  # Use our defined images directory
    
    try:
        predictor = DetectionPredictor(cfg)
        predictor()
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    predict()
    