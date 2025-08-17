import sys
import onnxruntime
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    with open('preprocessor.json', 'r') as f:
        preprocess_config = json.load(f)
    return config, preprocess_config

def preprocess_frame(frame, preprocess_config):
    """Preprocess video frame with proper handling for Instagram story aspect ratio"""
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get preprocessing parameters
    target_size = preprocess_config['pad_size']  # Should be 640
    rescale_factor = preprocess_config['rescale_factor']
    
    # Resize maintaining aspect ratio
    height, width = img.shape[:2]
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (center padding)
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    dx = (target_size - new_width) // 2
    dy = (target_size - new_height) // 2
    padded[dy:dy+new_height, dx:dx+new_width] = resized
    
    # Convert to float32 and normalize
    input_tensor = padded.astype(np.float32) * rescale_factor
    
    # Transpose from HWC to CHW format and add batch dimension
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, 0)
    
    return input_tensor, (scale, (dx, dy))

def postprocess_output(output, frame_shape, preprocessing_info, conf_threshold=0.3, iou_threshold=0.45):
    """Postprocess model output for 3-class fine-tuned model"""
    scale, (dx, dy) = preprocessing_info
    
    # Custom class mapping for fine-tuned model
    class_names = {
        0: "dj-air3",
        1: "uav", 
        2: "UAV"
    }
    
    # Output format: [batch, num_classes + 4, num_boxes]
    output = output[0]  # Remove batch dimension: [7, 8400]
    output = output.transpose()  # Convert to [8400, 7]
    
    # Extract boxes and scores
    boxes = output[:, :4]  # x_center, y_center, width, height
    class_scores = output[:, 4:]  # 3 class scores
    
    # Convert from center format to corner format
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Get best class for each detection
    best_class_ids = np.argmax(class_scores, axis=1)
    best_scores = np.max(class_scores, axis=1)
    
    # Filter based on confidence threshold
    conf_mask = best_scores > conf_threshold
    boxes = boxes[conf_mask]
    best_scores = best_scores[conf_mask]
    best_class_ids = best_class_ids[conf_mask]
    
    if len(boxes) == 0:
        return []
    
    # Scale back to original frame
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
    
    # Clamp boxes to frame boundaries
    boxes[:, 0] = np.clip(boxes[:, 0], 0, frame_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, frame_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, frame_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, frame_shape[0])  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), best_scores.tolist(), conf_threshold, iou_threshold)
    
    results = []
    
    if len(indices) > 0:
        for idx in indices:
            box = boxes[idx]
            score = best_scores[idx]
            class_id = best_class_ids[idx]
            label = class_names[class_id]
            
            results.append({
                'box': box.tolist(),
                'score': float(score),
                'label': label
            })
    
    return results

def draw_results(frame, results):
    """Draw bounding boxes and labels on frame"""
    # Define colors for each class
    colors = {
        "dj-air3": (0, 255, 0),      # Green
        "uav": (255, 0, 0),          # Blue  
        "UAV": (0, 0, 255)           # Red
    }
    
    for result in results:
        box = result['box']
        label = f"{result['label']} {result['score']:.2f}"
        color = colors.get(result['label'], (255, 255, 255))
        
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def setup_onnx_model(modelPath: str) -> tuple[str, onnxruntime.InferenceSession]:
    # Initialize ONNX Runtime session with fine-tuned model
    session = onnxruntime.InferenceSession(modelPath, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    return input_name, session

def process_video(video: Path, input_name: str, session: onnxruntime.InferenceSession, onnx_model_path: str) -> Path:
    """Main video processing function"""
    # Load configurations
    config, preprocess_config = load_config()
    
    print(f"Using fine-tuned model: {onnx_model_path}")
    print("Model classes: dj-air3, uav, something_else")

    # Open input video
    cap = cv2.VideoCapture(videoInputFile)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Aspect ratio: {width/height:.2f}")
    print("")
    
    # Setup output video writer
    output_path = f'{videoInputFile}_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections_count = 0

    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)
        
            # Preprocess frame
            input_tensor, preprocessing_info = preprocess_frame(frame, preprocess_config)
            
            # Run inference
            outputs = session.run(None, {input_name: input_tensor})
            output = outputs[0]
            
            # Post-process results (70% confidence threshold)
            results = postprocess_output(output, frame.shape, preprocessing_info, conf_threshold=0.2)
            
            if results:
                i = 1
                detections_count += len(results)
                # print(f"Frame {frame_count}: Found {len(results)} detections")
                for result in results:
                    # print(f"  - {result['label']}: {result['score']:.3f}")
                    i+=1
            
            # Draw results on frame
            frame_with_detections = draw_results(frame, results)
            
            # Write frame to output video
            out.write(frame_with_detections)
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
        
        # Clean up
        cap.release()
        out.release()
    
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Total detections: {detections_count}")
    return output_path

if __name__ == "__main__":
    videoInputFile = sys.argv[1]
    if (videoInputFile == ""):
        print("Missing video .mp4 file")
        exit(1)

    onnx_model_path = 'yolov11n-UAV-finetune.onnx'

    input_name, session = setup_onnx_model(onnx_model_path)
    output_path = process_video(videoInputFile, input_name, session, onnx_model_path)
    print(f"Output saved to: {output_path}")
