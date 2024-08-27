from pathlib import Path

from setuptools import glob

from ultralytics import YOLO
import numpy as np
from PIL import Image


def is_point_inside_polygon(x, y, polygon):
    """
    检查点是否在多边形内部
    参考：https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
                (x < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (
                        polygon[j][1] - polygon[i][1])):
            inside = not inside
        j = i
    return inside


def find_polygon_pixels(masks_xy, boxes_cls):  # 所有掩码像素点及其对应类别属性的列表
    # 初始化存储所有像素点和类别属性的列表
    all_pixels_with_cls = []

    # 遍历每个多边形
    for i, polygon in enumerate(masks_xy):
        cls = boxes_cls[i]  # 当前多边形的类别属性

        # 将浮点数坐标点转换为整数类型
        polygon = [(int(point[0]), int(point[1])) for point in polygon]

        # 找出当前多边形的边界框
        min_x = min(point[0] for point in polygon)
        max_x = max(point[0] for point in polygon)
        min_y = min(point[1] for point in polygon)
        max_y = max(point[1] for point in polygon)

        # 在边界框内遍历所有像素点
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # 检查像素点是否在多边形内部
                if is_point_inside_polygon(x, y, polygon):
                    # 将像素点坐标和类别属性组合成元组，添加到列表中
                    all_pixels_with_cls.append(((x, y), cls))

    return all_pixels_with_cls


def reconstruct_image(image_size, pixels_with_cls):
    # 创建一个和图片原始大小相同的黑色图像
    reconstructed_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # 将属性为 0 的像素点设为绿色，属性为 1 的像素点设为蓝色 ，其余的像素点默认为背景设为黑色
    for pixel, cls in pixels_with_cls:
        if cls == 0:
            reconstructed_image[pixel[1], pixel[0]] = [0, 255, 0]  # 绿色
        elif cls == 1:
            reconstructed_image[pixel[1], pixel[0]] = [0, 0, 255]  # 蓝色
        else:
            reconstructed_image[pixel[1], pixel[0]] = [0, 0, 0]  # 黑色

    return reconstructed_image


def predict_seg(image_path, model_path):
    """
    预测yoloV8分割
    :param image_path: 测试集合路径
    :param model_path: 模型路径
    """
    model = YOLO(model_path)

    files = glob.glob(image_path + "/*.png")
    for file in files:
        results = model(file)
        image = Image.open(file)
        file_name = Path(file).stem
        path_result = Path(image_path) / 'results'
        path_mask = Path(image_path) / 'masks'
        if not path_result.exists():
            path_result.mkdir()
        if not path_mask.exists():
            path_mask.mkdir()

        for result in results:
            boxes = result.boxes  # 输出的检测框
            masks = result.masks  # 输出的掩码信息

        masks_xy = masks.xy  # 每个掩码的边缘点坐标
        boxes_cls = boxes.cls  # 每个多边形的类别属性

        # 调用函数找出每个多边形内部的点和相应的类别属性
        try:
            all_pixels_with_cls = find_polygon_pixels(masks_xy, boxes_cls)
            image_size = image.size

            # print("所有像素点和相应的类别属性：", all_pixels_with_cls)  # 在终端显示所有掩码对应的坐标以及对应的属性元组

            reconstructed_image = reconstruct_image(image_size, all_pixels_with_cls)  # 重建图像
            Image.fromarray(reconstructed_image).save('{}/{}.png'.format(path_mask, file_name))  # 保存图像

            # Show the results
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.save('{}/{}.png'.format(path_result, file_name))  # save image
        except Exception as e:
            pass


if __name__ == '__main__':
    predict_seg('/home/pkc/AJ/2024/datasets/2407/v8seg/val/val',
                '/home/pkc/AJ/2024/pythonprojects/v8seg/hair_root/train4/weights/best.pt')