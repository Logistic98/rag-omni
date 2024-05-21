#!/bin/bash

docker build -t 'nginx_balance_image' .
docker run -itd --name nginx_balance -h nginx_balance -p 5000:5000 nginx_balance_image
docker update nginx_balance --restart=always