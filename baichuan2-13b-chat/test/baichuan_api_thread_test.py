# -*- coding: utf-8 -*-

from typing import Union, Any, List, Tuple
import requests
import json
import threading
import time


class Baichuan:
    def __init__(self, url):
        self.url = url

    def send_request(self, messages: List[dict]) -> Tuple[bool, Union[str, Any], float]:
        start_time = time.time()
        try:
            data = {"messages": messages}
            response = requests.post(self.url, json=data)
            response = json.loads(response.content)
            success = True
        except Exception as e:
            response = str(e)
            success = False
        end_time = time.time()
        return success, response, end_time - start_time


def worker(url, messages, index, stats):
    bc = Baichuan(url)
    success, response, duration = bc.send_request(messages)
    with stats['lock']:
        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1

        stats['total_duration'] += duration
        if duration < stats['min_duration']:
            stats['min_duration'] = duration
        if duration > stats['max_duration']:
            stats['max_duration'] = duration

    print(f"Thread {index}: {'Success' if success else 'Failure'}, Response: {response}, Duration: {duration}s")


if __name__ == '__main__':
    url = "http://127.0.0.1:1707/"
    messages = [{
        "role": "user",
        "content": "解释一下量子计算"
    }]

    num_threads = 3  # 测试并发线程数
    num_rounds = 3  # 测试轮数

    stats = {
        'success_count': 0,
        'failure_count': 0,
        'total_duration': 0.0,
        'min_duration': float('inf'),
        'max_duration': float('-inf'),
        'lock': threading.Lock()
    }

    for round in range(num_rounds):
        print(f"开始测试轮数: {round + 1}")
        threads = []

        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(url, messages, i, stats))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    # 输出总体统计结果
    avg_duration = stats['total_duration'] / (num_threads * num_rounds) if num_threads > 0 else 0
    print(f"总成功次数: {stats['success_count']}, 总失败次数: {stats['failure_count']}")
    print(f"整体最短耗时: {stats['min_duration']:.2f}s, 整体最长耗时: {stats['max_duration']:.2f}s, 整体平均耗时: {avg_duration:.2f}s")