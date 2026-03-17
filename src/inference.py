import json
import time

import cv2
from PIL import Image
import torch
from src.data.preprocess import decode
from src.utils.utils import calculate_metrics


def inference_yolo_crnn(
    yolo_det,
    img_path,
    conf_threshold=0.3,
    device="cpu",
    idx_to_char=None,
    crnn_transform=None,
    crnn_inference=None,
):
    """Inference using YOLO + CRNN pipeline"""
    predictions = []
    
    # YOLO detection 
    results = yolo_det(img_path, verbose=False, conf=conf_threshold)
    detections = json.loads(results[0].to_json())
    
    img = cv2.imread(img_path)
    
    for det in detections:
        if det["confidence"] < conf_threshold:
            continue
    
        # Get bounding box
        box = det['box']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        # Crop text region
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        
        # Convert to PIL image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        # CRNN recognition
        img_tensor = crnn_transform(cropped_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = crnn_inference(img_tensor)
            pred_text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)[0]
        
        predictions.append({
            'bbox': (x1, y1, x2 - x1, y2 - y1),
            'text': pred_text,
            'confidence': det['confidence']
        })
        
    return predictions
        

def inference_yolo_trocr(img_path, conf_threshold=0.3, yolo_det=None, device="cpu", trocr_processor=None, trocr_model=None):
    """Inference using YOLO + TrOCR pipeline"""

    predictions = []
    
    # YOLO Detection
    results = yolo_det(img_path, verbose=False, conf=conf_threshold)
    detections = json.loads(results[0].to_json())
    
    img = cv2.imread(img_path)
    
    for det in detections:
        if det['confidence'] < conf_threshold:
            continue
        
        # Get bounding box
        box = det['box']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        # Crop text region
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        
        # Convert to PIL image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
         
        # TrOCR Recognition
        pixel_values = trocr_processor(cropped_pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            pred_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        predictions.append({
            'bbox': (x1, y1, x2-x1, y2-y1),
            'text': pred_text,
            'confidence': det['confidence']
        })

    return predictions


def inference_easyocr(img_path, conf_threshold=0.3, easyocr_reader=None):
    """Inference using EasyOCR (end-to-end)"""
    
    predictions = []
    
    # EasyOCR Detection + Recognition 
    results = easyocr_reader.readtext(img_path)
    
    for result in results:
        bbox, text, confidence = result
        
        if confidence < conf_threshold:
            continue
        
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        
        predictions.append({
            'bbox': (x1, y1, x2-x1, y2-y1),
            'text': text,
            'confidence': confidence
        })

    return predictions


def match_predictions_to_ground_truth(predictions, ground_truths, iou_threshold=0.3):
    """Match predictions to ground truth using IoU and return matched pairs"""
    matched_preds = []
    matched_gts = []
    
    for gt in ground_truths:
        gt_bbox = gt['bbox']
        best_iou = 0
        best_pred = None 
        
        for pred in predictions:
            pred_bbox = pred['bbox']
            
            # Calculate IoU
            x1_gt, y1_gt, w_gt, h_gt = gt_bbox
            x1_pred, y1_pred, w_pred, h_pred = pred_bbox
            
            x1_inter = max(x1_gt, x1_pred)
            y1_inter = max(y1_gt, y1_pred)
            x2_inter = min(x1_gt + w_gt, x1_pred + w_pred)
            y2_inter = min(y1_gt + h_gt, y1_pred + h_pred)
            
            if x2_inter > x1_inter and y2_inter > y1_inter:
                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                area_gt = w_gt * h_gt
                area_pred = w_pred * h_pred
                union_area = area_gt + area_pred - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
                    
        if best_iou >= iou_threshold and best_pred:
            matched_preds.append(best_pred['text'])
            matched_gts.append(gt['label'])
            
    return matched_preds, matched_gts


def evaluate_model(inference_func, test_samples, model_name, conf_threshold=0.3):
    """Evaluate a model on test samples"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}...")
    print(f"{'='*60}")
    
    all_predictions = []
    all_ground_truths = []
    total_time = 0
    processed_images = 0
    
    images_dict = {}
    for item in test_samples:
        img_path = item['image_path']
        if img_path not in images_dict:
            images_dict[img_path] = []
        images_dict[img_path].append(item)
        
    for img_path, gt_items in images_dict.items():
        try:
            start_time = time.time()
            predictions = inference_func(img_path, conf_threshold=conf_threshold)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            processed_images += 1
            
            # Match predictions to ground truth
            matched_preds, matched_gts = match_predictions_to_ground_truth(
                predictions, gt_items, iou_threshold=0.3
            )
            
            all_predictions.extend(matched_preds)
            all_ground_truths.extend(matched_gts)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
        
    # Calculate metrics
    if len(all_predictions) > 0:
        metrics = calculate_metrics(all_predictions, all_ground_truths)
        avg_time = total_time / processed_images if processed_images > 0 else 0
        
        print(f"\nResults for {model_name}:")
        print(f"  - Character Accuracy: {metrics['char_accuracy']:.2f}%")
        print(f"  - Word Accuracy: {metrics['word_accuracy']:.2f}%")
        print(f"  - Average Time per Image: {avg_time:.4f}s")
        print(f"  - Total Images Processed: {processed_images}")
        print(f"  - Total Text Regions Matched: {len(all_predictions)}")
        
        return {
            'model': model_name,
            'char_acc': metrics['char_accuracy'],
            'word_acc': metrics['word_accuracy'],
            'avg_time': avg_time,
            'total_images': processed_images,
            'matched_regions': len(all_predictions)
        }
    else:
        print(f"No predictions matched for {model_name}")
        return None