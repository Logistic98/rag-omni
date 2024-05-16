# -*- coding: utf-8 -*-


class ResponseCode(object):
    SUCCESS = 200
    PARAM_FAIL = 400
    BUSINESS_FAIL = 500


class ResponseMessage(object):
    SUCCESS = "请求成功"
    PARAM_FAIL = "参数校验失败"
    BUSINESS_FAIL = "业务处理失败"