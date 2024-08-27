import os

from ultralytics import YOLO
from PIL import Image
import cv2

from ultralytics.utils.plot_cfg import plot_cfg

class PredictTools:
    """
    推理工具
    """

    def __init__(self, model_path, input_size):
        """
        创建实例
        """
        if os.path.exists(model_path) is False:
            raise 'not found {}'.format(model_path)
        if input_size is None:
            print('use default image size')
        self.model = YOLO(model_path)
        self.pred_array = None  # 保存预测结果
        self.images_path = None  # 推理图像路径

    def predict_pose(self, img_path):
        """
        预测单张图像
        :param img_path: 推理图像文件路径
        """
        self.pred_array = self.model(img_path)
        self.images_path = img_path

    def predict_batch(self, source, conf=0.25, iou=0.7, half=False, device=None, save=False, save_txt=False,
                           save_conf=False, plot_label=True, plot_box=True, plot_mask=True):
        """
        批推理推向，参数详见/doc/zh/models/predict.md
        :param source: 推理图像源目录
        :param conf: 置信度阈值
        :param iou: IOU阈值
        :param half: 是否使用fp16
        :param device: 推理设备
        :param save: 保存推理的图像
        :param save_txt: 保存推理结果
        :param save_conf: 保存带置信度的结果
        """
        plot_cfg(plot_label, plot_box, plot_mask)

        self.model.predict(source=source,
                           conf=conf,
                           iou=iou,
                           half=half,
                           device=device,
                           save=save,
                           save_txt=save_txt,
                           save_conf=save_conf)

    def predict_video(self, video_path, conf=0.8, iou=0.7):
        """
        视频推理
        Args:
            video_path: 视频地址
            conf: 置信度
            iou: IOU阈值
        Returns:

        """
        if not os.path.exists(video_path):
            raise FileNotFoundError('{} does not exist!'.format(video_path))
        results = self.model.predict(source=video_path, conf=conf, iou=iou, save=True)
        print(results)

        # # 打开视频
        # cap = cv2.VideoCapture(video_path)
        #
        # while cap.isOpened():
        #     # 获取图像
        #     res, frame = cap.read()
        #     # 如果读取成功
        #     if res:
        #         # 正向推理
        #         results = model(frame)
        #
        #         # 绘制结果
        #         annotated_frame = results[0].plot()
        #
        #         # 显示图像
        #         cv2.imshow(winname="YOLOV8", mat=annotated_frame)
        #
        #         # 按ESC退出
        #         if cv2.waitKey(1) == 27:
        #             break
        #
        #     else:
        #         break
        #
        # # 释放链接
        # cap.release()
        # # 销毁所有窗口
        # cv2.destroyAllWindows()


    def show(self):
        """
        可视化推理结果
        """
        if self.pred_array is None:
            raise 'here is not predicted'
        for p in self.pred_array:
            im_array = p.plot()
            im = Image.fromarray(im_array)
            im.show()


if __name__ == '__main__':
    pt = PredictTools(model_path='/home/pkc/AJ/2024/pythonprojects/v8seg/hair_root/train/weights/best.pt',
                      input_size=None)

    pt.predict_batch('/home/pkc/AJ/2024/pythonprojects/v8seg/run', save=True,
                     plot_label=False, plot_box=False)