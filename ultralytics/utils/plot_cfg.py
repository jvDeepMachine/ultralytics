import yaml

plot_label = False
plot_box = True
plot_mask = True


def label_cfg():
    return plot_label


def box_cfg():
    return plot_box


def mask_cfg():
    return plot_mask


def plot_cfg(label=True, box=True, mask=True):
    global plot_label, plot_box, plot_mask  # 声明为全局变量
    plot_label = label
    plot_box = box
    plot_mask = mask
