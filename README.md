# captcha 12306

识别 12306 验证码

## usage

1. 下载模型

在 [AI Studio 数据集](https://aistudio.baidu.com/aistudio/datasetdetail/22010)，下载 text.tar.gz 和 image.tar.gz 到项目根目录，解压

```
$ mkdir model
$ tar xzvf text.tar.gz -C model
$ tar xzvf image.tar.gz -C model
```

目录结构为

```
model
├── image
│   ├── best_model.h5
│   ├── label_encoder.pkl
│   ├── logs.json
│   └── logs.png
└── text
    ├── best_model.h5
    ├── label_encoder.pkl
    ├── logs.json
    └── logs.png
```

2. 识别验证码

指定图片识别

```
$ python parse.py -t model/text -i model/image -c demo.jpg
```

![demo.jpg](demo.jpg)

```
text label:
['鞭炮']
image labels:
['鞭炮' '路灯' '安全帽' '蜥蜴' '安全帽' '蜜蜂' '蜡烛' '安全帽']
```

不使用 `-c`，可从官网实时下载一张验证码识别

```
$ python parse.py -t model/text -i model/image
```

## how it works

如何采集数据，标注数据到训练模型，:point_right: [参见文档](./doc/how.md) :point_left:

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
