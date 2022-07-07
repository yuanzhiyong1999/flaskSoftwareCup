import base64
import os
import traceback

import cv2
import numpy as np
from PIL import Image
from cv2 import waitKey
from flask import Flask, request
import paddlers as pdrs
from skimage import img_as_ubyte
import urllib
from paddlers.tasks.utils.visualize import visualize_detection
import unifiedResponse
from skimage.io import imread, imsave, show

import utils
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)


def visualize_attention_map(attention_map):
    """
    The attention map is a matrix ranging from 0 to 1, where the greater the value,
    the greater attention is suggests.
    :param attention_map: np.numpy matrix hanging from 0 to 1
    :return np.array matrix with rang [0, 255]
    """
    attention_map_color = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1], 3],
        dtype=np.uint8
    )

    red_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8
    ) + 255

    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)
    #
    blue_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8
    ) + 255

    blue_color_map = blue_color_map * (0.2 - attention_map)
    blue_color_map = np.array(blue_color_map, dtype=np.uint8)

    attention_map_color[:, :, 2] = red_color_map
    attention_map_color[:, :, 0] = blue_color_map

    return attention_map_color


def img_handle_other(result, task_name, cover_img):
    # 对检测结果处理
    path1 = utils.get_path_name(task_name) + '-result.png'
    imsave(path1, result['label_map'], check_contrast=False)
    url1 = utils.uploadPic(path1)
    utils.delete_temp_pic(path1)

    # 对热力图处理
    path2 = utils.get_path_name(task_name) + '-attention_img.png'
    score = result['score_map'][:, :, 1]
    score = visualize_attention_map(score)
    attention_img = score + cover_img
    imsave(path2, attention_img, check_contrast=False)
    url2 = utils.uploadPic(path2)
    utils.delete_temp_pic(path2)
    return url1, url2


# 图片增强
@app.route("/enhancement", methods=["POST"])
def img_enhancement():
    rJ = request.json

    url = rJ["value"]
    params = rJ["params"]

    # 获取image
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # 进行操作
    if params["blur"] != 0:
        blur = params["blur"]
        if blur % 2 == 0:
            blur += 1
        image = cv2.GaussianBlur(image, (blur, blur), 0)

    if params["brightness"] != 0:
        image = params["brightness"]

    if params["hue"] != 0:
        hue = params["hue"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 0] += hue
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    if params["saturation"] != 0:
        saturation = params["saturation"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] = saturation * image[:, :, 1]
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    if params["thresh_slider"] != 0:
        thresh_slider = params["thresh_slider"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        image = cv2.threshold(image, thresh_slider, 255, cv2.THRESH_BINARY)[1]

    if params["canny_slider_a"] != 0 | params["canny_slider_b"] != 0:
        canny_slider_a = params["canny_slider_a"]
        canny_slider_b = params["canny_slider_b"]
        image = cv2.Canny(image, canny_slider_a, canny_slider_b)

    if params['contour_slider'] != 0:
        contour_slider = params['contour_slider']
        base_slider = params['base_slider']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.GaussianBlur(image, (21, 21), 1)
        image = cv2.inRange(image, np.array([contour_slider, base_slider, 40]),
                            np.array([contour_slider + 30, 255, 220]))
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)

    # 转为BASA64
    result = cv2.imencode('.png', image)[1]
    result = str(base64.b64encode(result))[2:-1]
    return unifiedResponse.result(data={"value": result})


# 目标提取
te = pdrs.deploy.Predictor('model/inference_model/target_extraction')


@app.route('/target_extraction', methods=['POST'])
def target_extraction():
    image_url = request.form.get('image')
    if image_url is not None:
        resp = urllib.request.urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        utils.judge_path('target_extraction')
        try:
            result = te.predict(img_file=image)
            url1, url2 = img_handle_other(result, 'target_extraction', image)
            return unifiedResponse.result(data={'binary_img': url1, 'attention_img': url2})
        except Exception as e:
            print(traceback.format_exc())
            return unifiedResponse.error('图片预测失败')
    else:
        return unifiedResponse.error('图片为空')


# 变化检测
cd = pdrs.deploy.Predictor('model/inference_model/change_detection')


# result = predictor.predict(img_file=('img/change_detection/2022/05/24/train_1.png',
#                                      'img/change_detection/2022/05/24/train_1 (1).png'),
#                            warmup_iters=30,
#                            repeats=30)

@app.route('/change_detection', methods=['POST'])
def change_detection():
    before_url = request.form.get('before')
    after_url = request.form.get('after')
    if before_url is not None and after_url is not None:
        resp1 = urllib.request.urlopen(before_url)
        resp2 = urllib.request.urlopen(after_url)
        before = np.asarray(bytearray(resp1.read()), dtype="uint8")
        after = np.asarray(bytearray(resp2.read()), dtype="uint8")
        before = cv2.imdecode(before, cv2.IMREAD_COLOR)
        after = cv2.imdecode(after, cv2.IMREAD_COLOR)
        utils.judge_path('change_detection')
        try:
            result = cd.predict(img_file=(before, after))
            url1, url2 = img_handle_other(result[0], 'change_detection', before)

            return unifiedResponse.result(data={'binary_img': url1, 'attention_img': url2})
        except Exception as e:
            print(traceback.format_exc())
            return unifiedResponse.error('图片预测失败')
    else:
        return unifiedResponse.error('图片为空')


# 目标检测
td = pdrs.deploy.Predictor('model/inference_model/target_detection')


# @app.route('/test', methods=['POST'])
# def test():
#     try:
#         image_url = "http://cup.lijx.cloud/img/target_detection/2022/07/07/overpass_4.jpg"
#         resp = urllib.request.urlopen(image_url)
#         image = np.asarray(bytearray(resp.read()), dtype="uint8")
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         utils.judge_path('target_detection')
#         result = td.predict(img_file=image)
#         vis = image
#         vis = visualize_detection(
#             vis, result,
#             color=np.asarray([[0, 255, 0]], dtype=np.uint8),
#             threshold=0.2, save_dir=None
#         )
#         path = utils.get_path_name('target_detection') + '-result.png'
#         imsave(path, vis, check_contrast=False)
#         url = utils.uploadPic(path)
#         utils.delete_temp_pic(path)
#         cv2.imshow('a', vis)
#         cv2.waitKey(0)
#
#         return unifiedResponse.result(data=url)
#     except Exception as e:
#         print(traceback.format_exc())
#         return unifiedResponse.error('图片预测失败')


@app.route('/target_detection', methods=['POST'])
def target_detection():
    image_url = request.form.get('image')
    if image_url is not None:
        resp = urllib.request.urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        utils.judge_path('target_detection')
        try:
            result = td.predict(img_file=image)
            vis = image
            vis = visualize_detection(
                vis, result,
                color=np.asarray([[0, 255, 0]], dtype=np.uint8),
                threshold=0.2, save_dir=None
            )
            path = utils.get_path_name('target_detection') + '-result.png'
            imsave(path, vis, check_contrast=False)
            url = utils.uploadPic(path)
            utils.delete_temp_pic(path)

            return unifiedResponse.result(data=url)
        except Exception as e:
            print(traceback.format_exc())
            return unifiedResponse.error('图片预测失败')
    else:
        return unifiedResponse.error('图片为空')


# 地物分类
tc = pdrs.deploy.Predictor('model/inference_model/terrain_classification')


@app.route('/terrain_classification', methods=['POST'])
def terrain_classification():
    image_url = request.form.get('image')
    if image_url is not None:
        resp = urllib.request.urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        utils.judge_path('terrain_classification')
        try:
            result = tc.predict(img_file=image)
            url1, url2 = img_handle_other(result, 'terrain_classification', image)

            return unifiedResponse.result(data={'binary_img': url1, 'attention_img': url2})

        except Exception as e:
            print(traceback.format_exc())
            return unifiedResponse.error('图片预测失败')
    else:
        return unifiedResponse.error('图片为空')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
