# -*- coding: utf-8 -*-

import requests


def retrieval_test(url, params):
    r = requests.get(url, params=params)
    print(r.text)


if __name__ == '__main__':
    url = 'http://{0}:{1}/api/rag/retrieval'.format("127.0.0.1", "5001")
    params = {'question': "国务院对于地方政府性债务管理的意见", 'top_k': 3}
    retrieval_test(url, params)


