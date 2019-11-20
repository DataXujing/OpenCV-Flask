'''
author(s): xujing from Medcare
date: 2019-03-20

flask调用opencv并基于yolo-lite做目标检测。
解决了：
    在html中嵌入opencv视频流
    opencv putText的中文显示
    darknet调用yolo-lite
    多线程，多点同时访问
    ajax异步更新echarts的json数据，实时绘制识别结果！

问题： yolo-lite在no-GPU下的识别FPS没有做到paper中说的那么高！
'''

from flask import Response
from flask import Flask
from flask import render_template

import os
import uuid
import threading
import argparse
from ctypes import *

import math
import random
import numpy as np
import configparser

import imutils
import cv2
from imutils.video import VideoStream
from PIL import Image,ImageDraw,ImageFont
import matplotlib.cm as mpcm

import datetime
import time

from pyecharts.charts import Bar
from pyecharts import options as opts

app = Flask(__name__)

outputFrame = None
temp_str = str(uuid.uuid1())
print(temp_str)
lock = threading.Lock()

config = configparser.ConfigParser()
config.read('config.ini')
if config['IPCapture']['IP'] != 'no':
    # vs = VideoStream(src= config['IPCapture']['IP']).start()  # ip摄像头
    vs = cv2.VideoCapture(config['IPCapture']['IP']) # ip摄像头
elif config['USBCapture']['USB'] != 'no':
    # vs = VideoStream(src=0).start()  # USB摄像头或采集卡设备
    vs = cv2.VideoCapture(0)  # USB摄像头或采集卡设备
elif config['PiCamera']['PI'] != 'no':
    # vs = VideoStream(usePiCamera=1).start()  # 树莓派
    vs = cv2.VideoCapture(1) # 树莓派
elif  config['VideoPath']['PATH'] != 'no':
    # vs = VideoStream(src="test.mp4").start()  # 本地视频源
    vs = cv2.VideoCapture("test.mp4") # 本地视频源

hasGPU = config['Device']['Device']


label_name = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
probs = [0.0] * len(label_name)

time.sleep(2.0)

# ---------------------------通过darknet调用yolo-lite--------------------------------
def change_cv2_draw(image,strs,local,sizes,colour):
    '''解决openCV putText中文显示问题
    '''
    cv2img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("./static/fonts/Microsoft-Yahei-UI-Light.ttc",sizes,encoding='utf-8')
    draw.text(local,strs,colour,font=font)
    image = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)

    return image

def colors_subselect(colors, num_classes=20):
    '''颜色映射
    '''
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors = colors_subselect(mpcm.plasma.colors, num_classes=20)
colors_tableau = [(255, 152, 150),(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
                 (255, 152, 150),(148, 103, 189), (197, 176, 213)]

# 调用darknet需要的一些方法
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']

    if hasGPU == "True":
        winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")  # GPU!
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    else:
        winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_no_gpu.dll")
        lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)  # Lunix

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):

    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


netMain = None
metaMain = None
altNames = None

def performDetect(imagePath="test.jpg", thresh=0.5, configPath="./model/tiny-yolov2-trial13-noBatch.cfg", weightPath="./model/tiny-yolov2-trial13_noBatch.weights", metaPath="./model/voc.data", showImage=True, makeImageOnly=False, initOnly=False):

    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    #detections = detect(netMain, metaMain, imagePath, thresh)  # if is used cv2.imread(image)
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)

    if showImage:
        try:
            scale = 0.4
            text_thickness = 1
            line_type = 8
            thickness=2

            image = cv2.imread(imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            imcaption = []
            img_prob = [0.0]*len(label_name)

            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                img_prob[label_name.index(label)] = np.rint(100 * confidence)

                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape

                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)

                color = colors_tableau[label_name.index(detection[0])]
                p1 = (xCoord, yCoord)
                p2 = (xCoord + xEntent,yCoord + yExtent)
                if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
                    continue

                cv2.rectangle(image, p1, p2, color, thickness)

                text_size, baseline = cv2.getTextSize(pstring, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
                cv2.rectangle(image, (p1[0], p1[1] - thickness*10 - baseline), (p1[0] + 2*(text_size[0]-20), p1[1]), color, -1)
                image = change_cv2_draw(image,pstring,(p1[0],p1[1]-7*baseline),20,(255,255,255))
       
        except Exception as e:
            print("Unable to show image: "+str(e))
    return image, img_prob


#--------------------------falsk调用OpenCV和YOLO-lite------------------------------------

# index视图函数
@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


def detect_yolo_lite():
    '''
    调用yolo-lite
    '''
    global vs, outputFrame, lock, probs
    total = 0

    while True:
        ret,frame = vs.read()
        total += 1
        # frame = imutils.resize(frame, width=400)
        
        # if total/10 == 0:
        save_path = "./static/images/"+ temp_str + ".jpg"
        cv2.imwrite(save_path,frame)
        frame, probs = performDetect(imagePath=save_path)
        # print(frame)

        with lock:  # 多线程的线程锁，确保当前线程的数据不被其他线程修改！
            outputFrame = frame


def generate():
    '''构建生成器
    '''

    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag,encodedImage) = cv2.imencode(".jpg",outputFrame)

            if not flag:
                continue
        yield(b"--frame\r\n" b"Content-Type:image/jpeg\r\n\r\n"+bytearray(encodedImage)+b"\r\n")

# 显示帧
@app.route("/video_feed")
def video_feed():
    return Response(generate(),mimetype='multipart/x-mixed-replace;boundary=frame')


# ajax异步更新echarts数据
@app.route("/get_bar")
def get_bar():
    
    global probs
    bar = (
      Bar()
        .add_xaxis(label_name)
        .add_yaxis("Detection Probs",probs)
    )
    # print(bar.render_embed())
    # print(bar.dump_options())
    # return render_template("index.html",bar_data=bar.dump_options())
    # return bar.dump_options_with_quotes()
    return bar.dump_options()




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--ip",type=str,required=True,help="IP")
    ap.add_argument("-o","--port",type=int,required=True,help="port")
   
    args = vars(ap.parse_args())
    
    # 多线程
    t = threading.Thread(target=detect_yolo_lite)
    t.daemon = True
    t.start()

    app.run(host=args["ip"],port=args["port"],debug=True,threaded=True,
        use_reloader=False)


#release视频流
vs.stop()


