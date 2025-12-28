from ultralytics import YOLO

def main():
    # Using yolov8s.pt (Small) - Better for reading text on signs than Nano
    model = YOLO('yolov8s.pt')  

    model.train(
        data=r"C:\Users\aksha\OneDrive\Desktop\bhosdi_2\data.yaml", 
        epochs=50,                   
        imgsz=640,                   
        batch=16,
        device=0,                    # Change to 'cpu' if you don't have a GPU
        name='speed_limit_model'     # Results will save to runs/detect/speed_limit_model
    )

if __name__ == '__main__':
    main()