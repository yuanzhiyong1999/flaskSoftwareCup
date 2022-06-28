import os
import traceback

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request
import paddlers as pdrs
from skimage import img_as_ubyte

import unifiedResponse
from skimage.io import imread, imsave

import utils

app = Flask(__name__)

# 目标提取
te = pdrs.deploy.Predictor('model/inference_model/target_extraction')


@app.route('/target_extraction', methods=['POST'])
def target_extraction():
    image = request.files.get('image')
    if image is not None:
        utils.judge_path('target_extraction')
        try:
            result = te.predict(img_file=np.array(Image.open(image)))
            path = utils.get_path_name('target_extraction') + '-result.png'
            imsave(path, result['label_map'], check_contrast=False)
            url = utils.uploadPic(path)
            utils.delete_temp_pic(path)

            return unifiedResponse.result(data={'url': url})
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
    before = request.files.get('before')
    after = request.files.get('after')
    if before is not None and after is not None:
        utils.judge_path('change_detection')
        try:
            result = cd.predict(img_file=(np.array(Image.open(before)), np.array(Image.open(after))))

            path = utils.get_path_name('change_detection') + '-result.png'
            imsave(path, result[0]['label_map'], check_contrast=False)
            url = utils.uploadPic(path)
            utils.delete_temp_pic(path)

            return unifiedResponse.result(data={'url': url})
        except Exception as e:
            print(traceback.format_exc())
            return unifiedResponse.error('图片预测失败')
    else:
        return unifiedResponse.error('图片为空')


# 目标检测
td = pdrs.deploy.Predictor('model/inference_model/target_detection')


@app.route('/target_detection', methods=['POST'])
def target_detection():
    image = request.files.get('image')
    if image is not None:
        utils.judge_path('target_detection')
        try:
            result = td.predict(img_file=np.array(Image.open(image)))
            path = utils.get_path_name('target_detection') + '-result.png'
            imsave(path, result['label_map'], check_contrast=False)
            url = utils.uploadPic(path)
            utils.delete_temp_pic(path)

            return unifiedResponse.result(data={'url': url})
        except Exception as e:
            print(traceback.format_exc())
            return unifiedResponse.error('图片预测失败')
    else:
        return unifiedResponse.error('图片为空')


# 地物分类
tc = pdrs.deploy.Predictor('model/inference_model/terrain_classification')


@app.route('/terrain_classification', methods=['POST'])
def terrain_classification():
    image = request.files.get('image')
    if image is not None:
        utils.judge_path('terrain_classification')
        try:
            result = tc.predict(img_file=np.array(Image.open(image)))
            path = utils.get_path_name('terrain_classification') + '-result.png'
            imsave(path, result['label_map'], check_contrast=False)
            url = utils.uploadPic(path)
            utils.delete_temp_pic(path)

            return unifiedResponse.result(data={'url': url})
        except Exception as e:
            print(traceback.format_exc())
            return unifiedResponse.error('图片预测失败')
    else:
        return unifiedResponse.error('图片为空')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
