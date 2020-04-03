# captcha 12306

识别 12306 验证码

## usage

1. 下载模型



2. 识别验证码

```
$ python parse.py -t model/text -i model/image
```

## how it works

[参见文档](./doc/how.md)

## env

```
keras==2.3.1
tensorflow-gpu=2.1.0
opencv-contrib-python==3.4.2.17
numpy==1.18.2
imutils==0.5.3
scikit-learn==0.22.2
matplotlib==3.2.1
```
