from ultralytics import YOLO

def train_yolov8():
    # Load YOLOv8 model (use larger model if needed)
    model = YOLO('yolov8n.pt')  # or 'yolov8m.pt' or 'yolov8s.pt' for better accuracy

    # Train the model
    results = model.train(
        data='data.yaml',    # Path to corrected data.yaml
        epochs=5,          # Increased to 200 epochs for better generalization
        batch=8,             # Lowered batch size to prevent memory issues
        imgsz=640,           # Input image size
        patience=5,          # Early stopping after 5 epochs of no improvement
        lr0=0.005,           # Lower initial learning rate for smoother training
        lrf=0.01,            # Final learning rate
        weight_decay=0.0005, # Regularization to prevent overfitting
        optimizer='AdamW',   # AdamW is better for small datasets
        augment=True,        # Enable data augmentation
        hsv_h=0.005,         # Lower hue augmentation to avoid color confusion
        hsv_s=0.4,           # Lower saturation augmentation
        hsv_v=0.2,           # Lower value augmentation
        flipud=0.5,          # Vertical flip
        fliplr=0.5,          # Horizontal flip
        degrees=10,          # Random rotation
        translate=0.1,       # Random translation
        shear=2.0,           # Random shear
        perspective=0.0,     # No perspective change
        warmup_epochs=5,     # Increased warmup for more stable gradients
        cos_lr=True,         # Cosine learning rate decay
        close_mosaic=10,     # Close mosaic after 10 epochs for stability
        cache=True,          # Cache dataset for faster training
        workers=4,           # Use multi-threading for faster loading
        iou=0.4,             # Lower IOU threshold for better recall
        conf=0.55            # Confidence threshold based on F1 curve
    )

if __name__ == "__main__":
    train_yolov8()
