"""
@FileName ：utils.py
@Author ：Zhiyong Yuan
@Date ：2022/5/20 9:28 
@Tools ：PyCharm
@Description：各种小工具
"""
import datetime
import os
import uuid

import qiniu
from qiniu import put_data, put_file

QINIU_ACCESS_KEY = 'ZNW9hy971YuWSQOFHusyTQwGgRIJhY9JbRO6iuqf'
QINIU_SECRET_KEY = 'S1B26pdzPlsCoAqtwFxPjmvteOuSRBfJrhAh6AuX'
QINIU_BASE_URL = 'http://cup.lijx.cloud/'
QINIU_BUCKET = 'softwarecup'


def get_path(task_name):
    now = datetime.datetime.now()
    new_path = "img/{0}/{1}/".format(task_name, now.strftime("%Y/%m/%d"))
    return new_path


def get_name():
    new_name = '{0}'.format(str(uuid.uuid4()).replace('-', ''))
    return new_name


def get_path_name(task_name):
    now = datetime.datetime.now()
    new_name = "img/{0}/{1}/{2}".format(task_name, now.strftime("%Y/%m/%d"), str(uuid.uuid4()).replace('-', ''))
    return new_name


def judge_path(task_name):
    path = get_path(task_name)
    if not os.path.exists(path):
        os.makedirs(path)


def delete_temp_pic(path):
    os.remove(path)  # 删除文件

# 上传到七牛云
def uploadPic(path):
    q = qiniu.Auth(QINIU_ACCESS_KEY, QINIU_SECRET_KEY)
    token = q.upload_token(QINIU_BUCKET)

    ret, info = put_file(token, path, path)
    if info.status_code == 200:
        url = '%s%s' % (QINIU_BASE_URL, ret['key'])
    else:
        return None
    return url
