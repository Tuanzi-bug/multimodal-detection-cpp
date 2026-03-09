import time
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.io as scio
from yolo import YOLO
from eFUMI_VCA_initialize import normalize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import colorsys

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#  ACE高光谱目标检测算法
def ace_tarsig(x, double_arrow_s, mu, sqrtDxinvU):
    x = np.float64(x)
    centered_x = x - np.tile(mu, (1, x.shape[1]))

    arrow_x = sqrtDxinvU.dot(centered_x)
    fro_arrow_x = np.linalg.norm(arrow_x, axis=0, ord=2)
    double_arrow_x = arrow_x / fro_arrow_x

    Y = double_arrow_s.dot(double_arrow_x)
    return Y


def read_first_column_from_txt(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            first_column = [line.strip().split()[0] for line in lines]
            return first_column
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None


def Multimode_collaboration(map_out_path, img_name, hyimg):

    # 对于图像img_name，获取第一阶段Yolo对其检测出的类别predict_classes
    with open(os.path.join(map_out_path, "detection-results/"+img_name+".txt"), 'r') as f:
        data = f.readlines()
        max_val = float('-inf')
        max_char = ''
        for line in data:
            line = line.strip().split()
            if len(line) >= 2:
                try:
                    val = float(line[1])
                    if val > max_val:
                        max_val = val
                        max_char = line[0]
                except ValueError:
                    pass
        predict_classes = [max_char]
    # predict_classes = read_first_column_from_txt(predict_box_path + img_name + '.txt')
    # predict_classes = list(set(predict_classes))

    # 高光谱图像标准化
    test_data = np.float64(hyimg)
    test_data_ = np.reshape(np.transpose(test_data, (2, 1, 0)), (test_data.shape[2], -1))
    test_norm = normalize(test_data_, 1)

    # 基于第一阶段Yolo检测出图像中可能包含的目标类别，对其配准高光谱图像进行
    # 该类别下的ACE高光谱目标检测
    ace_detect_result = {}
    norm_result = []
    for predict_class in predict_classes:
        spectral_lib_list = os.listdir("./spectral_lib/" + predict_class)
        map_list = []
        # for s in sig_list:
        for gt_spec in spectral_lib_list:
            print("calculate " + predict_class + ": " + gt_spec + "...")

            gt_sig = scio.loadmat(os.path.join("./spectral_lib/" + predict_class, gt_spec))['data']
            mu_x_minus = gt_sig[0, :].reshape(1, -1)
            du = gt_sig[1:-1, :]
            sig = gt_sig[-1, :].reshape(1, -1)
            confid_temp = ace_tarsig(test_norm, sig, mu_x_minus.T, du)

            map_ = confid_temp.reshape((test_data.shape[1], test_data.shape[0])).T
            map_ = (map_ - np.min(map_)) / (np.max(map_) - np.min(map_))
            map_list.append(map_)
        map_list = np.array(map_list)

        # mean_map为对于某一类别的高光谱检测结果图
        mean_map = np.max(map_list, axis=0)
        # mean_map = (mean_map - np.min(mean_map)) / (np.max(mean_map) - np.min(mean_map))
        # norm_result.append(mean_map)
        ace_detect_result[predict_class] = mean_map
        print("done.")

    # norm_result = np.array(norm_result)
    # norm_result = (norm_result - np.min(norm_result)) / (np.max(norm_result) - np.min(norm_result))

    for predict_class in predict_classes:
        # 绘制高光谱图像对于某类别的最初检测结果
        print(predict_class)
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(ace_detect_result[predict_class], cmap='nipy_spectral',
                   vmax=1, vmin=0)
        plt.colorbar()
        plt.axis('off')
        plt.title(predict_class + " spectrum detection results")
        # plt.show()
        plt.imsave(os.path.join(map_out_path, 'temp-results/Ace_'+img_name+'.png'),
                   ace_detect_result[predict_class], cmap='nipy_spectral')
        plt.close(fig)

        # 对检测结果进行膨胀操作
        kernel1 = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(ace_detect_result[predict_class], kernel1, iterations=1)
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(dilated_image, cmap='nipy_spectral', vmax=1, vmin=0)
        plt.colorbar()
        plt.axis('off')
        plt.title(predict_class + " spectrum detection results")
        # plt.show()
        plt.imsave(os.path.join(map_out_path, 'temp-results/dilatedAce_' + img_name + '.png'),
                   dilated_image, cmap='nipy_spectral')
        plt.close(fig)

        # 对膨胀的检测结果进行增强对比度操作
        adjusted_image = np.clip(dilated_image * 1.3, 0, 1)
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(adjusted_image, cmap='nipy_spectral', vmax=1, vmin=0)
        plt.colorbar()
        plt.axis('off')
        plt.title(predict_class + " spectrum detection results")
        # plt.show()
        plt.imsave(os.path.join(map_out_path, 'temp-results/enhanceAce_' + img_name + '.png'),
                   adjusted_image, cmap='nipy_spectral')
        plt.close(fig)

        # 最终的高光谱检测结果图
        ace_detect_result[predict_class] = adjusted_image


    out_boxes = []
    out_scores = []
    out_classes = []
    voc_classes = {'ship': 0, 'aircraft': 1, 'roof': 2, 'car': 3, 'oiltank': 4}
    class_names = ['ship', 'aircraft', 'roof', 'car', 'oiltank']

    # 多模协同筛选第一阶段预测框
    with open(os.path.join(map_out_path, "multimode-detection-results/"+img_name+".txt"), "w") as file_:
        with open(os.path.join(map_out_path, "detection-results/"+img_name+".txt"), "r") as f:
            for line_ in f.readlines():
                line = line_.strip('\n')
                line = line.split()
                boxclass = line[0]
                if boxclass == max_char:
                    x1 = np.maximum(int(line[2]), 0)
                    y1 = np.maximum(int(line[3]), 0)
                    x2 = np.maximum(int(line[4]), 0)
                    y2 = np.maximum(int(line[5]), 0)

                    temp1 = ace_detect_result[predict_class][y1:y2 + 1, x1:x2 + 1]
                    # count_above_threshold = np.sum(temp1 > 0.8)
                    # total_elements = temp1.size
                    # if count_above_threshold / total_elements >= 0.3:
                    if (np.max(temp1[len(temp1) // 2, len(temp1[0]) // 3: 2 * len(temp1[0]) // 3]) >= 0.80) and \
                            (np.max(temp1[len(temp1) // 3: 2 * len(temp1) // 3, len(temp1[0])//2]) >= 0.80):
                        out_boxes.append([y1, x1, y2, x2])
                        out_scores.append(float(line[1]))
                        out_classes.append(int(voc_classes[line[0]]))
                        file_.write(line_)
                else:
                    continue
    # 多模协调检测框
    out_boxes_ = np.array(out_boxes)
    out_scores_ = np.array(out_scores)
    out_classes_ = np.array(out_classes)
    out_boxes_ = tf.convert_to_tensor(out_boxes_)
    out_scores_ = tf.convert_to_tensor(out_scores_)
    out_classes_ = tf.convert_to_tensor(out_classes_)

    # 绘制保存多模协调目标检测图像
    image = Image.open("VOCdevkit/VOC2007/JPEGImages/" + img_name + ".jpg")
    print("Draw multi-modal object detection results...")
    print('Found {} boxes for {}'.format(len(out_boxes_), 'img'))
    # ---------------------------------------------------------#
    #   设置字体与边框厚度
    # ---------------------------------------------------------#
    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // 640.0, 1))

    hsv_tuples = [(x / 5, 1., 1.) for x in range(5)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    # ---------------------------------------------------------#
    #   图像绘制
    # ---------------------------------------------------------#
    for i, c in list(enumerate(out_classes_)):
        predicted_class = class_names[int(c)]
        box = out_boxes_[i]
        score = out_scores_[i]
        top, left, bottom, right = box

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    if not os.path.exists("./test/multimodal-predict"):
        os.makedirs("./test/multimodal-predict")
    image.save("./test/multimodal-predict/"+img_name+".png", quality=95, subsampling=0)
    # plt.imshow(image)
    # plt.show()



