from ultralytics import YOLO

def test_model():
    model = YOLO('C:/Users/Kutsa/Documents/GitHub/Capstone/runs\detect/train/weights/best.pt')  # Path to the best trained model
    results = model.val(data='data.yaml')

if __name__ == "__main__":
    test_model()
