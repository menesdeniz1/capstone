from ultralytics import YOLO

def test_model():
    model = YOLO('runs/train/weights/best.pt')  # Path to the best trained model
    results = model.val(data='data.yaml')

if __name__ == "__main__":
    test_model()
