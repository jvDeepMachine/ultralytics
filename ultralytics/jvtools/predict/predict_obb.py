from ultralytics import YOLO
from ultralytics.utils.plot_cfg import plot_cfg

# load model
model = YOLO('/home/pkc/AJ/2024/pythonprojects/ultralytics/weights/best.pt')

# eval
plot_label = False
plot_box = False
plot_mask = True
plot_cfg(plot_label, plot_box, plot_mask)
result = model.predict('/home/pkc/AJ/2024/develop/quickTest/assets/wound/after.png',
                       save=True)
