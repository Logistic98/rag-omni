#!/bin/bash

base_path=$(cd `dirname $0`; pwd)
input_path="${base_path}/input"
output_path="${base_path}/output"

docker build -t marker-image .                                  
docker run -itd --name marker -v ${input_path}:/code/input -v ${output_path}:/code/output marker-image:latest  
docker update marker --restart=always                           