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


def result(code=HttpCode.ok, message='', data=None, kwargs=None):
    json_dict = {'code': code, 'message': message, 'data': data}

    if kwargs and isinstance(kwargs, dict) and kwargs.keys():
        json_dict.update(kwargs)

    return jsonify(json_dict)


def ok():
    return result()


def error(message=''):
    return result(code=HttpCode.error, message=message)


def params_error(message='', data=None):
    return result(code=HttpCode.paramserror, message=message, data=data)


def unauth(message='', data=None):
    return result(code=HttpCode.unauth, message=message, data=data)


def method_error(message='', data=None):
    return result(code=HttpCode.methoderror, message=message, data=data)


def server_error(message='', data=None):
    return result(code=HttpCode.servererror, message=message, data=data)
