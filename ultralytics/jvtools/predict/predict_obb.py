from ultralytics import YOLO

# load model
model = YOLO('/home/pkc/AJ/2024/pythonprojects/ultralytics/weights/best.pt')

# eval
result = model.predict('/home/pkc/AJ/2024/develop/quickTest/assets/wound/after.png',
                       save=True)
