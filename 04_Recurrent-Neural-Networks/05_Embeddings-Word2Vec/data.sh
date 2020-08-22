#!/bin/bash

mkdir -p data

wget https://s3.amazonaws.com/video.udacity-data.com/topher/2018/October/5bbe6499_text8/text8.zip -O data/text8.zip
unzip data/text8.zip data/text8