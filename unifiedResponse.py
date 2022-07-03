"""
@FileName ：unifiedResponse.py
@Author ：Zhiyong Yuan
@Date ：2022/5/18 15:36 
@Tools ：PyCharm
@Description：统一结果封装
"""
from flask import jsonify


class HttpCode(object):
    """docstring for HttpCode"""
    ok = 200
    paramserror = 400
    unauth = 401
    methoderror = 405
    servererror = 500
    error = 444


def result(code=HttpCode.ok, msg='', data=None, kwargs=None):
    json_dict = {'code': code, 'msg': msg, 'data': data}

    if kwargs and isinstance(kwargs, dict) and kwargs.keys():
        json_dict.update(kwargs)

    return jsonify(json_dict)


def ok():
    return result()


def error(msg=''):
    return result(code=HttpCode.error, msg=msg)


def params_error(msg='', data=None):
    return result(code=HttpCode.paramserror, msg=msg, data=data)


def unauth(msg='', data=None):
    return result(code=HttpCode.unauth, msg=msg, data=data)


def method_error(msg='', data=None):
    return result(code=HttpCode.methoderror, msg=msg, data=data)


def server_error(msg='', data=None):
    return result(code=HttpCode.servererror, msg=msg, data=data)
