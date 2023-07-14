#!/usr/bin/python
# -*- coding: UTF-8 -*-
import glob
import multiprocessing
import random
import torch
import json
import tqdm
import cv2,os,pdb
import xml.etree.ElementTree as ET
import argparse
import sys
from pathlib import Path
from xml.dom import minidom
import requests, cv2, math
from urllib import request
from socket import *
import time
import shutil
import copy
import re
import configparser
from collections import defaultdict
import numpy as np
# import xlwt
# import xlrd
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import yaml

sys.path.append(str(Path.cwd().parent.parent.parent))
# def opt1_mean(sample):


import scipy.io as scio    # mat format

class multiprocess(object):
    def __init__(self,framegap = 100,bs = False):
        self.framegap = framegap
        self.configs = []
        self.bs = bs
    def video2img(self,config):
        index, config = config
        for videopath in config["video"]:
            cap = cv2.VideoCapture(videopath)  # 文件名及格式
            count = 0
            index = 1
            file = videopath.split(os.sep)[-1]
            dir = videopath.replace(file, '')
            filename = file.replace('.mp4','').replace('.MP4','').replace('.avi','')
            os.makedirs(config['saveimgs'] + filename + '/', exist_ok=True)
            # os.makedirs(config['saveimgs'] +'labels/' +filename + '/', exist_ok=True)
            ret, frame = cap.read()
            # import pdb
            # pdb.set_trace()
            bs = cv2.createBackgroundSubtractorMOG2(25 * 2, 16, False)
            while (ret):
                # capture frame-by-frame
                if self.bs:
                    frame = cv2.UMat(frame)
                    fg_mask = bs.apply(frame)
                    frame = bs.getBackgroundImage()
                if count % self.framegap == 0 :
                    namenum = str(count).zfill(8)
                    savepath = config['saveimgs'] + filename + '/'+ f"{filename}_{namenum}.jpg"  #weizhicuowu
                    if not os.path.isfile(savepath):
                        cv2.imwrite(savepath, frame)
                    index += 1
                count += 1
                ret, frame = cap.read()
                # if index > 10:
                #     break
            # when everything done , release the capture
            cap.release()
            # cv2.destroyAllWindows()
        return 0
    def forward(self):

        with torch.multiprocessing.Pool(self.multinum) as pool:
            m = list(tqdm.tqdm(pool.imap(self.video2img, self.configs[:self.multinum]), total=len(self.configs[:self.multinum]), position=0))
        # for config in configs[in_pool:]:
        #         self.video2img(config)
    def getconfig(self,videospath,saveimgs,multinum):
        self.multinum = multinum
        config_template = {
            'video':[],
            'saveimgs':'',

        }
        self.configs = []
        os.makedirs(saveimgs, exist_ok=True)
        if os.path.isdir(videospath):
            files = [video for video in walkfile(videospath)]
            for i in range(multinum):
                config = copy.deepcopy(config_template)
                config['saveimgs'] = saveimgs
                for x,file in enumerate(files):
                    if x % multinum == i:
                        if file.endswith('mp4')   or file.endswith('MP4') :
                            config['video'].append(file)

                # config['video'] = allvideo[i]
                self.configs.append([i, config])
        elif os.path.isfile(videospath):
            config = copy.deepcopy(config_template)
            config['saveimgs'] = saveimgs
            config['video'].append(videospath)
            self.configs.append([0, config])
        # import pdb
        # pdb.set_trace()
        return self.configs

class VMCI(object):
    """通信类"""

    def Sendtojava(self, request_url, json_data):
        """调用api"""
        try:
            req_url = request_url
            r = requests.post(url=req_url, json=json_data)
            print('数据是否发送： ', r.text)
        except request.HTTPError:
            # print("there is an error")
            pass  # 跳过错误，不进行处理，直接继续执行

    def Clienttcp(self, HOST, PORT, BUFSIZ, Senddata):
        """TCP 客户机"""
        ADDR = (HOST, PORT)
        tcpCliSock = socket(AF_INET, SOCK_STREAM)
        tcpCliSock.connect(ADDR)
        tcpCliSock.send(Senddata.encode())
        Img_src = tcpCliSock.recv(BUFSIZ).decode()
        # if not Img_src:
        # continue
        # time.sleep(1)
        tcpCliSock.close()
        print('\n')

    def writetxt(seif, bad, m):  # 写入txt接口到f：\zzh.txt
        if bad[0] == '正常片':
            s = 0
        else:
            s = 1
        t = m
        try:
            try:
                txt = open('E:\\zzh.txt', 'w+')
                n = str(t) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
            except:
                time.sleep(0.05)
                txt = open('E:\\zzh.txt', 'w+')
                n = str(t) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
        except:
            time.sleep(0.05)
            txt = open('E:\\zzh.txt', 'w+')
            n = str(t) + '\n' + str(s)
            txt.write(n)
            txt.close()
            print('xieru:', m, n)

    def writehalm(self, bad, m):  # halm软件写入脚本
        if bad[0] == '正常片':
            s = 0
        else:
            s = 1
        # print(m)
        name = m.split('.')[0]
        realname = name.split('_')
        try:
            try:
                txt = open('E:\\zzh.txt', 'w+')
                n = str(realname[1]) + '\n' + str(realname[2]) + '\n' + str(realname[3]) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
            except:
                time.sleep(0.05)
                txt = open('E:\\zzh.txt', 'w+')
                n = str(realname[1]) + '\n' + str(realname[2]) + '\n' + str(realname[3]) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
        except:
            time.sleep(0.05)
            txt = open('E:\\zzh.txt', 'w+')
            n = str(realname[1]) + '\n' + str(realname[2]) + '\n' + str(realname[3]) + '\n' + str(s)
            txt.write(n)
            txt.close()
            print('xieru:', m, n)

class writexml(object):
    """预测的标签数据重新写入xml"""

    def writetoxml(self, gz, save_path):
        """
        :param gz: ['10943', 520, 520, 1, ['断栅', 194, 13, 263, 27], ['断栅', 69, 166, 138, 194], ['断栅', 374, 41, 429, 69]]
        :return: 新生成的xml
        """
        dom = minidom.Document()  # 1.创建DOM树对象

        root_node = dom.createElement('annotation')  # 2.创建根节点。每次都要用DOM对象来创建任何节点。

        root_node.setAttribute('verified', 'no')  # 3.设置根节点的属性

        dom.appendChild(root_node)  # 4.用DOM对象添加根节点

        # 用DOM对象创建元素子节点
        # folder
        folder_node = dom.createElement('folder')
        root_node.appendChild(folder_node)  # 用父节点对象添加元素子节点
        folder_text = dom.createTextNode('预测图片做标签')  # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        folder_node.appendChild(folder_text)  # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点

        # filename
        filename_node = dom.createElement('filename')
        root_node.appendChild(filename_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        filename_text = dom.createTextNode(str(gz[0]))  # gz[0]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        filename_node.appendChild(filename_text)

        # path
        path_node = dom.createElement('path')
        root_node.appendChild(path_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        path_text = dom.createTextNode('默认路径')
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        path_node.appendChild(path_text)

        # source
        source_node = dom.createElement('source')
        root_node.appendChild(source_node)
        # database
        database_node = dom.createElement('database')
        source_node.appendChild(database_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        database_text = dom.createTextNode('Unknown')
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        database_node.appendChild(database_text)

        # size
        size_node = dom.createElement('size')
        root_node.appendChild(size_node)
        # width
        width_node = dom.createElement('width')
        size_node.appendChild(width_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        width_text = dom.createTextNode(str(gz[1]))  # gz[1]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        width_node.appendChild(width_text)
        # height
        height_node = dom.createElement('height')
        size_node.appendChild(height_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        height_text = dom.createTextNode(str(gz[2]))  # gz[2]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        height_node.appendChild(height_text)
        # depth
        depth_node = dom.createElement('depth')
        size_node.appendChild(depth_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        depth_text = dom.createTextNode(str(gz[3]))  # gz[3]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        depth_node.appendChild(depth_text)

        # segmented
        segmented_node = dom.createElement('segmented')
        root_node.appendChild(segmented_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        segmented_text = dom.createTextNode('0')
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        segmented_node.appendChild(segmented_text)

        length = len(gz)  # 求数据输入长度
        for i in range(length):
            if i < 4:  # 从第5个开始
                continue
            # object
            object_node = dom.createElement('object')
            root_node.appendChild(object_node)
            # name
            name_node = dom.createElement('name')
            object_node.appendChild(name_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            name_text = dom.createTextNode(str(gz[i][0]))  # gz[i][0]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            name_node.appendChild(name_text)
            # pose
            pose_node = dom.createElement('pose')
            object_node.appendChild(pose_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            pose_text = dom.createTextNode('Unspecified')
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            pose_node.appendChild(pose_text)
            # truncated
            truncated_node = dom.createElement('truncated')
            object_node.appendChild(truncated_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            truncated_text = dom.createTextNode('0')
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            truncated_node.appendChild(truncated_text)
            # Difficult
            Difficult_node = dom.createElement('Difficult')
            object_node.appendChild(Difficult_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            Difficult_text = dom.createTextNode('0')
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            Difficult_node.appendChild(Difficult_text)
            # bndbox
            bnd_node = dom.createElement('bndbox')
            object_node.appendChild(bnd_node)
            # xmin
            xmin_node = dom.createElement('xmin')
            bnd_node.appendChild(xmin_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            xmin_text = dom.createTextNode(str(gz[i][1]))  # gz[i][1]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            xmin_node.appendChild(xmin_text)
            # ymin
            ymin_node = dom.createElement('ymin')
            bnd_node.appendChild(ymin_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            ymin_text = dom.createTextNode(str(gz[i][2]))  # gz[i][2]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            ymin_node.appendChild(ymin_text)
            # xmax
            xmax_node = dom.createElement('xmax')
            bnd_node.appendChild(xmax_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            xmax_text = dom.createTextNode(str(gz[i][3]))  # gz[i][3]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            xmax_node.appendChild(xmax_text)
            # ymax
            ymax_node = dom.createElement('ymax')
            bnd_node.appendChild(ymax_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            ymax_text = dom.createTextNode(str(gz[i][4]))  # gz[i][4]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            ymax_node.appendChild(ymax_text)

        # 每一个结点对象（包括dom对象本身）都有输出XML内容的方法，如：toxml()--字符串, toprettyxml()--美化树形格式。

        try:
            with open(save_path + '%s.xml' % (str(gz[0])), 'w', encoding='UTF-8') as fh:
                # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
                # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
                dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')
                # print('写入xml OK!')
        except Exception as err:
            print('错误信息：{0}'.format(err))

class Judge_EL(object):

    def ELsummary(self, jsonObj):

        """接收{"黑斑黑点": "1", "水印": "1", "划痕": "10"}，返回根据规则判断的列表"""
        ELclassify = []
        for key in jsonObj.keys():
            if key == "同心圆发黑":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            # elif key == "黑边":
            # if int(jsonObj[key]) > 0:
            # ELclassify.append(key)
            elif key == "隐裂":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "断栅":
                if int(jsonObj[key]) > 6:
                    ELclassify.append(key)
            elif key == "连续断栅":
                if int(jsonObj[key]) > 0:
                    ELclassify.append("断栅")
            elif key == "雾状发黑":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "大划痕":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            # elif key == "划痕":
            #    if int(jsonObj[key]) > 2:
            #        ELclassify.append(key)
            elif key == "小划痕":
                if int(jsonObj[key]) > 7:
                    ELclassify.append(key)
            # elif key == "吸球印吸盘印":
            #    if int(jsonObj[key]) > 0:
            #        ELclassify.append(key)
            elif key == "气流片":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "水印手指印":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "皮带印滚轮印":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
        return ELclassify

    def transform_form(self, thresh, transform_type, scale):
        # 输入图片和模式，输出图片
        """开、闭运算操作"""
        kern = int((scale + 0.2) * 3)
        if transform_type == 'closing':  # 闭运算：先膨胀再腐蚀
            dilation_kernel = np.ones((kern, kern), np.uint8)  # 增加白色区域
            dilation = cv2.dilate(thresh, dilation_kernel, iterations=1)
            erode_kernel = np.ones((kern, kern), np.uint8)
            closing_image = cv2.erode(dilation, erode_kernel, iterations=1)  # 腐蚀 增加黑色区域
            return closing_image
        elif transform_type == 'opening':  # 开运算；先腐蚀再膨胀
            erode_kernel = np.ones((kern, kern), np.uint8)
            # erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kern, kern))
            erode = cv2.erode(thresh, erode_kernel, iterations=1)  # 腐蚀 增加黑色区域
            dilation_kernel = np.ones((kern, kern), np.uint8)
            # dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern, kern))
            opening_image = cv2.dilate(erode, dilation_kernel, iterations=2)
            return opening_image

    def TFhbhd(self, img, zone, dingdian):
        """
        :param img: 图片，直接读取的图片，未经过灰度转化，如果输入灰度图，下面的灰度转化语句需要注释掉
        :param zone: 数组，四通道[[xmin,xmax,ymin,ymax]]
        :return: 'y'指判断黑斑黑点，'n'判断非黑斑黑点
        """

        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 求输入图片背景灰度

        imgraycopy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xd1 = dingdian[0][0]
        yd1 = dingdian[0][1]
        xd2 = dingdian[1][0]
        yd2 = dingdian[1][1]
        xd3 = dingdian[2][0]
        yd3 = dingdian[2][1]
        xd4 = dingdian[3][0]
        yd4 = dingdian[3][1]
        xd5 = dingdian[4][0]
        yd5 = dingdian[4][1]
        xd6 = dingdian[5][0]
        yd6 = dingdian[5][1]
        xd7 = dingdian[6][0]
        yd7 = dingdian[6][1]
        xd8 = dingdian[7][0]
        yd8 = dingdian[7][1]
        k1 = (yd2 - yd1) / (xd2 - xd1)
        k2 = (yd3 - yd2) / (xd3 - xd2)
        k3 = (yd4 - yd3) / (xd4 - xd3)
        k4 = (xd5 - xd4) / (yd5 - yd4)
        k5 = (yd6 - yd5) / (xd6 - xd5)
        k6 = (yd7 - yd6) / (xd7 - xd6)
        k7 = (yd8 - yd7) / (xd8 - xd7)
        k8 = (xd1 - xd8) / (yd1 - yd8)
        '''
        imgray1 = abs(imgraycopy - 127)
        w2, h2 = imgray.shape
        overallarea, overallgray = 0, 0
        scale = w2 / 520
        for x in range(0, w2, 2):
            for y in range(0, h2, 2):
                if imgray1[x][y] < 80:
                    overallarea += 1
                    overallgray += imgray[x][y]
        overallagray = overallgray / (overallarea + 1)
        print(scale, overallagray)
        # 根据求得的背景灰度agray，将图片整体灰度调整，规避灰度越界，标准灰度163
        for x1 in range(w2):
            #if max(imgray[x1]) > 255:
            for y1 in range(h2):
                    if imgray[x1][y1] * (163 / overallagray) > 255:
                        imgray[x1][y1] = 255
                    else:
                        imgray[x1][y1] * (163 / overallagray)
        '''
        w2, h2 = imgray.shape
        overallarea, overallgray = 0, 0
        scale = w2 / 520
        # 根据调整后的图片和标记区间zone（数组），求每个区间的属性并计算属于那种黑斑黑点
        qiandian, shendian, qianban, shenban, median = 0, 0, 0, 0, 0
        for p in range(len(zone)):
            image = imgray[zone[p][1]:zone[p][3], zone[p][0]:zone[p][2]]
            w, h = image.shape
            # image_show('image', image)
            if scale > 2:
                w1 = 33
            else:
                w1 = 19
            th = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, w1,
                                       3)  # 自适应均值阈值并黑白反转
            th1 = th[zone[p][1]:zone[p][3], zone[p][0]:zone[p][2]]
            if zone[p][0] > scale * 400 or zone[p][0] < scale * 100 or zone[p][1] > scale * 400 or zone[p][
                1] < scale * 100:
                thres = 88
                threshold = 80
            else:
                thres = 93
                threshold = 85
            ret, th2 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
            # image_show('th', th)
            image1 = self.transform_form(th1, 'opening', scale)
            image2 = self.transform_form(th2, 'opening', scale)
            # show('open', image1)
            # image_show('guding', image2)

            partarea1, partgray1, partarea2, partgray2 = 0, 0, 0, 0
            for x in range(w):
                for y in range(h):
                    ys = zone[p][0] + y
                    xs = zone[p][1] + x
                    if image1[x][y] == 255 and 160 > image[x][y] > 20:
                        if 0 < ys < 43 or 72 < ys < 152 or 169 < ys < 253 or 268 < ys < 355 or 372 < ys < 453 or 476 < ys < 520:
                            if k1 * (xs - xd1) + yd1 <= ys <= k5 * (xs - xd5) + yd5 and k2 * (
                                    xs - xd2) + yd2 <= ys <= k6 * (xs - xd6) + yd6 and k3 * (
                                    xs - xd3) + yd3 <= ys <= k7 * (xs - xd7) + yd7 and k8 * (
                                    ys - yd8) + xd8 <= xs <= k4 * (ys - yd4) + xd4:
                                partarea1 += 1
                                partgray1 += image[x][y]
            partagray = partgray1 / (partarea1 + 1)
            for x2 in range(w):
                for y2 in range(h):
                    ys = zone[p][0] + y2
                    xs = zone[p][1] + x2
                    if image2[x2][y2] == 255 and 160 > image[x2][y2] > 20:
                        if 0 < ys < 43 or 72 < ys < 152 or 169 < ys < 253 or 268 < ys < 355 or 372 < ys < 453 or 476 < ys < 520:
                            if k1 * (xs - xd1) + yd1 <= ys <= k5 * (xs - xd5) + yd5 and k2 * (
                                    xs - xd2) + yd2 <= ys <= k6 * (xs - xd6) + yd6 and k3 * (
                                    xs - xd3) + yd3 <= ys <= k7 * (xs - xd7) + yd7 and k8 * (
                                    ys - yd8) + xd8 <= xs <= k4 * (ys - yd4) + xd4:
                                partarea2 += 1
                                partgray2 += image[x2][y2]
            partagray2 = partgray2 / (partarea2 + 1)
            if partarea2 - partarea1 > scale * 150:
                partarea1 = (2 * partarea2 + partarea1) / 3
                partagray = partagray2

            print(partagray, partarea1)
            if partagray > thres and partarea1 < scale * 210:
                qiandian += 1
            elif partagray > thres + 15 and partarea1 > scale * 210:
                qianban += 1
            elif partagray < thres and partarea1 < scale * 210 and partarea1 > scale * 40:
                shendian += 1
            elif partagray < thres and partarea1 < scale * 210 and partarea1 < scale * 40:
                qiandian += 1
            elif partagray < thres + 15 and partarea1 > scale * 210:
                shenban += 1
            if partarea1 > scale * 350:
                shenban += 1
            if 150 < partarea1 < 210 and partagray < thres + 30:
                median += 1
                median = int(median / 2)
        if qiandian > 7 or qianban + shendian + median >= 2 or shenban > 0 or len(zone) > 7:
            print("浅点", qiandian, '浅斑', qianban, '深斑', shenban, '深点', shendian)
            return '黑斑黑点'
        else:
            return '未知'

    def yinlie(self, zone):
        '''

        :param zone:  [x,y,x,y,p,w,h]
        :return:
        '''
        final = []
        for k in zone:
            total_center = [k[6] / 2, k[5] / 2]
            center = [(k[2] + k[0]) / 2, (k[3] + k[1]) / 2]
            dis = np.sqrt((total_center[0] - center[0]) ** 2 + (total_center[1] - center[1]) ** 2)  # 欧式距离
            x1, y1, x2, y2, x3, y3 = total_center[0], total_center[1], k[6] / 2, 0, center[0], center[1]

            # 计算三条边长
            a = math.sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3))
            b = math.sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3))
            c = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

            # 利用余弦定理计算三个角的角度
            A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
            # B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
            # C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
            # 输出三个角的角度
            # print("There three angles are", round(A, 2), round(B, 2), round(C, 2))
            if (42 < A < 48 or 132 < A < 138) and dis > (k[5] * 0.5 + 40):  # 角度限定,距离限定
                if k[4] > 0.82:
                    # if k[4] > 0.82:  # 概率限定
                    final.append(k)
                else:
                    continue
            else:
                final.append(k)
        return final

class histogram(object):
    def histogram(self, dst, clahe_dir, file):
        """
        区域直方图均衡
        :param dst: 原始图片路径
        :param clahe_dir: 处理后的图片保存目录
        :return: 处理后的图片路径
        """
        img = cv2.imread(dst, 0)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(25, 25))
        cl1 = clahe.apply(img)
        clahe_el_path = os.path.join(clahe_dir, file)
        cv2.imwrite(clahe_el_path, cl1)  # 把bmp\jpg格式经过处理后都转化成jpg格式
        return clahe_el_path

def videow(img_vid_path):
    '''
    opencv_机器学习-图片合成视频
    实现步骤:
    1.加载视频
    2.读取视频的Info信息
    3.通过parse方法完成数据的解析拿到单帧视频
    4.imshow，imwrite展示和保存
    5.本地保存格式：image+数字。jpg        以第一张size为模板
    '''
    img = cv2.imread(os.path.join(img_vid_path, 'vehicle_0000385.jpg'))
    # 获取当前图片的信息
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    videowrite = cv2.VideoWriter(os.path.join(img_vid_path + 'test.mp4'), -1, 30, size)
    for i in range(385, 415):
        fileName = 'vehicle_0000' + str(i) + '.jpg'
        img = cv2.imread(os.path.join(img_vid_path, fileName))
        # 写入参数，参数是图片编码之前的数据
        videowrite.write(img)
    print('end!')
def video2img(videopath,gap=10):
    cap = cv2.VideoCapture(videopath)  # 文件名及格式
    count = 0
    index = 1
    file = videopath.split('\\')[-1]
    dir = videopath.replace(file,'')
    filename = file.split('.')[-2]
    os.makedirs(f'{dir}/{filename}', exist_ok=True)
    while (True):
        # capture frame-by-frame
        ret, frame = cap.read()
        if count % gap == 0:
            namenum =str(index).zfill(5)
            savepath = f'{dir}/{filename}/img_{namenum}.jpg'
            cv2.imwrite(savepath,frame)
            index+=1
        count += 1
        if index > 10:
            break
    # when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()
"""图片展示类"""

def image_show(barname, image):
    """
    :param barname: 图片显示的名称
    :param image: 输入
    :return:
    """
    cv2.namedWindow(barname, cv2.WINDOW_NORMAL)
    cv2.imshow(barname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def json_write(img_path,xml_path,label_path):
        annots = [os.path.join(xml_path, s) for s in os.listdir(xml_path)]  # 训练样本的xml路径
        json_list = []
        for annot in annots:
            """依次解析XML文件"""
            filename = annot.split('/')[-1]
            x = filename.split('.')[-1]
            filename_img = filename.replace(f'.{x}', '')
            #filename_img = filename.split('.')[-2]
            file_path = img_path + '/' + filename_img +'.jpg'
            print(file_path)
            et = ET.parse(annot)
            element = et.getroot()
            element_objs = element.findall('object')
            target_list = []
            for element_obj in element_objs:
                reframe = []
                #class_name = element_obj.find('name').text
                #if class_name == '小车':
                #    first = 0
                #else:
                #    first = 1
                reframe.append(int(element_obj.find('bndbox').find('xmin').text))
                reframe.append(int(element_obj.find('bndbox').find('ymin').text))
                reframe.append(int(element_obj.find('bndbox').find('xmax').text))
                reframe.append(int(element_obj.find('bndbox').find('ymax').text))
                trframe = xyxy2xywh_c(reframe)
                target_list.append(trframe)
            annot_dict = [('input', file_path), ('target', target_list)]
            dict1 = dict(annot_dict)
            json_list.append(dict1)
        #json1 = json.dumps(json_list)
        #print(json1)
        #with open('D:/pythonprojects/TrainSet/detection/v2.3train_test/label_test.json', 'w') as f:
        #    json.dump(json1, f)
        with open(label_path,'w',encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample,ensure_ascii=False) + "\n")

def json_write_bdd_old(img_path,json_old,label_path):
        annots = [os.path.join(json_old, s) for s in os.listdir(json_old)]  # 训练样本的xml路径
        json_list = []
        for annot in annots:
            """依次解析json文件"""
            filename = annot.split('/')[-1]
            filename_img = filename.split('.')[-2]
            file_path = img_path + '/' + filename_img +'.jpg'
            print('annot', annot , 'file_path',file_path)
            bboxs_bdd = 1,1
            #print(bboxs_bdd)
            if bboxs_bdd:
                annot_dict = [('input', file_path), ('target', bboxs_bdd)]
                dict1 = dict(annot_dict)
                json_list.append(dict1)
        with open(label_path, 'w', encoding='utf-8') as f:
                for sample in json_list:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def json_write_bdd(img_path,img_path_new, json_old, label_path):
        '''
        name -=[['daytime', 'night', 'dawn/dusk', 'undefined']
        :param img_path:
        :param json_old:
        :param label_path:
        :return:
        '''
        annots = [os.path.join(json_old, s) for s in os.listdir(json_old)]  # 训练样本的xml路径
        json_daytime_list = []
        json_night_list = []
        json_dawn_list = []
        for annot in annots:
            """依次解析json文件"""
            filename = annot.split('/')[-1]
            filename_img = filename.split('.')[-2]
            file_path = img_path + '/' + filename_img + '.jpg'
            #print('annot', annot, 'file_path', file_path)
            bboxs_bdd,name = 1,1
            # print(bboxs_bdd)
            if bboxs_bdd:
                annot_dict = [('input', file_path), ('target', bboxs_bdd)]
                dict1 = dict(annot_dict)
            if name == 'daytime' or name == 'undefined':
                json_daytime_list.append(dict1)
                newfile_path = img_path_new + '/' + 'daytime' + '/' + filename_img + '.jpg'
                shutil.copyfile(file_path,newfile_path)
            elif name == 'night':
                json_night_list.append(dict1)
                newfile_path = img_path_new + '/' + 'night' + '/' + filename_img + '.jpg'
                shutil.copyfile(file_path, newfile_path)
            else:
                json_dawn_list.append(dict1)
                newfile_path = img_path_new + '/' + 'dawn' + '/' + filename_img + '.jpg'
                shutil.copyfile(file_path, newfile_path)
        with open(label_path + '/' + 'label_bdd_daytime.json', 'w', encoding='utf-8') as f:
            for sample in json_daytime_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(label_path + '/' + 'label_bdd_night.json', 'w', encoding='utf-8') as f2:
            for sample in json_night_list:
                f2.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(label_path + '/' + 'label_bdd_dawn.json', 'w', encoding='utf-8') as f3:
            for sample in json_dawn_list:
                f3.write(json.dumps(sample, ensure_ascii=False) + "\n")

def check_json(json_label):
    outdebug = "/home/linwang/dataset/anomaly_test/outdebug/"
    os.makedirs(outdebug,exist_ok=True)
    f = open(json_label)
    lines = f.readlines()
    for i,line in tqdm.tqdm(enumerate(lines)):
        info = json.loads(line)
        if os.path.isfile(info['input']):
            img = cv2.imread(info['input'])
            for sample in info['target']:
                cv2.rectangle(img,(sample[0],sample[1]),(sample[2],sample[3]),(255,0,0),3)
            cv2.imwrite(outdebug + os.path.basename(info['input']),img)

def bddjson_yolo5_bm(labelpath = ''):
        parent = str(Path(labelpath).parent) + os.sep
        # labelpaths = "/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/labels/valids/fe1f55fa-19ba3600.txt"
        labelpaths = glob.glob(labelpath+'*.txt')
        newlabel = '/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/labels/newlabel.json'
        # imgpath = "/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/images/valids/fe1f55fa-19ba3600.jpg"
        json_list = []
        for i,labelpath in tqdm.tqdm(enumerate(labelpaths)):
            imgpath = labelpath.replace('labels','images')
            imgpath = imgpath.replace('txt', 'jpg')
            if os.path.isfile(imgpath):
                if i < 2:
                    img = cv2.imread(imgpath)
                    w,h,_ = img.shape
                with open(labelpath, 'r') as t:
                    target = []
                    t = t.read().splitlines()
                    for sample in t:
                        sample = sample.split(' ')
                        sample = np.array(list(map(float, sample)))
                        # import pdb
                        # pdb.set_trace()
                        sample[1:5] = sample[1:5]*[h,w,h,w]
                        sample = xywh_c2xyxy(sample[1:5],sample[0])
                        target.append(sample)

                annot_dict = [('input', imgpath), ('target', target)]
                dict1 = dict(annot_dict)
                json_list.append(dict1)
                        # cv2.rectangle(img,(sample[0],sample[1]),(sample[2],sample[3]),(0,0,255),2)
                    # cv2.imwrite('/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/inference/debug.jpg',img)
            else:
                continue
        with open(newlabel, 'w', encoding='utf-8') as f:
                for sample in json_list:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def load_data(datapath):
            dataset = []
            with open(datapath, "r") as fid:
                for line in fid:
                    sample = json.loads(line)
                    dataset.append(sample)
                    #print(type(sample))
            return dataset

def ua_data(inpath,imgdir,outpath,jsonpathc,jsonpathn,jsonpathr,jsonpaths):
        '''

        :param inpath: xml
        :imgpath :img
        :param outpath:  xml or json savepath
        :return:
        '''
        annots = [os.path.join(inpath, s) for s in os.listdir(inpath)]  # 训练样本的xml路径
        outimgnum = 1
        json_list_sunny = []
        json_list_rainy = []
        json_list_night = []
        json_list_cloudy = []
        for annot in annots:
            """依次解析XML文件"""
            et = ET.parse(annot)
            element = et.getroot()
            filename = annot.split('/')[-1]
            imgdirname = filename.replace('.xml','')
            #element_objs = element.findall('target')
            #tag = element.tag
            #attrib = element.attrib['name']  # zheshi tupianwenjianjia he xml mingzi
            #value = element.text
            idx = 1
            for child in element:
                bbox_list = []
                #print('1', child.attrib)
                for sec_child in child:
                    #print('2', sec_child.attrib)
                    for tri_child in sec_child:
                        #print('3' ,tri_child.tag, tri_child.attrib)
                        for four_child in tri_child:
                            if four_child.tag == 'box':
                                bbox = [0]
                                bbox.append(
                                    int(float(four_child.attrib['left'])) + int(float(four_child.attrib['width']) / 2))
                                bbox.append(
                                    int(float(four_child.attrib['top'])) + int(float(four_child.attrib['height']) / 2))
                                bbox.append(int(float(four_child.attrib['width'])))
                                bbox.append(int(float(four_child.attrib['height'])))
                                bbox_list.append(bbox)
                                #print('4', four_child.tag, bbox)
                if idx < 2:
                    weather = child.attrib.get('sence_weather')
                    idx += 1
                    out_img_path = outpath + '/' + weather
                    os.makedirs(out_img_path,exist_ok=True)
                print(bbox_list)
                imgpath = imgdir + '/' + imgdirname + '/' +'img' + str((child.attrib.get('num'))).zfill(5) + '.jpg'
                save_path = outpath + '/'  + weather + '/' + 'img'  + str(outimgnum).zfill(8) + '.jpg'
                outimgnum += 1
                print(imgpath , save_path)
                if bbox_list and outimgnum % 10 == 0:
                    shutil.copyfile(imgpath,save_path)
                    annot_dict = [('input', save_path), ('target', bbox_list)]
                    dict1 = dict(annot_dict)
                    if weather == 'cloudy':
                        json_list_cloudy.append(dict1)
                    elif weather == 'night':
                        json_list_night.append(dict1)
                    elif weather == 'rainy':
                        json_list_rainy.append(dict1)
                    else:
                        json_list_sunny.append(dict1)

        with open(jsonpathc, 'w', encoding='utf-8') as f:
                for sample in json_list_cloudy:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(jsonpathn, 'w', encoding='utf-8') as f:
            for sample in json_list_night:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(jsonpathr, 'w', encoding='utf-8') as f:
            for sample in json_list_rainy:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(jsonpaths, 'w', encoding='utf-8') as f:
            for sample in json_list_sunny:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ua_data_train(inpath,imgdir,outpath,jsonpath):
        '''

        :param inpath: xml
        :imgpath :img
        :param outpath:  xml or json savepath
        :return:
        '''
        annots = [os.path.join(inpath, s) for s in os.listdir(inpath)]  # 训练样本的xml路径
        outimgnum = 1
        json_list = []
        json_list_rainy = []
        json_list_night = []
        json_list_cloudy = []
        for annot in annots:
            """依次解析XML文件"""
            et = ET.parse(annot)
            element = et.getroot()
            filename = annot.split('/')[-1]
            imgdirname = filename.replace('.xml','')
            #element_objs = element.findall('target')
            #tag = element.tag
            #attrib = element.attrib['name']  # zheshi tupianwenjianjia he xml mingzi
            #value = element.text
            idx = 1
            for child in element:
                bbox_list = []
                #print('1', child.attrib)
                for sec_child in child:
                    #print('2', sec_child.attrib)
                    for tri_child in sec_child:
                        #print('3' ,tri_child.tag, tri_child.attrib)
                        for four_child in tri_child:
                            if four_child.tag == 'box':
                                bbox = [0]
                                bbox.append(
                                    int(float(four_child.attrib['left'])) + int(float(four_child.attrib['width']) / 2))
                                bbox.append(
                                    int(float(four_child.attrib['top'])) + int(float(four_child.attrib['height']) / 2))
                                bbox.append(int(float(four_child.attrib['width'])))
                                bbox.append(int(float(four_child.attrib['height'])))
                                bbox_list.append(bbox)
                                #print('4', four_child.tag, bbox)
                if idx < 2:
                    weather = child.attrib.get('sence_weather')
                    idx += 1
                    out_img_path = outpath + '/' + weather
                    os.makedirs(out_img_path,exist_ok=True)
                print(bbox_list)
                imgpath = imgdir + '/' + imgdirname + '/' +'img' + str((child.attrib.get('num'))).zfill(5) + '.jpg'
                save_path = outpath + '/'  + weather + '/' + 'img'  + str(outimgnum).zfill(8) + '.jpg'
                outimgnum += 1
                print(imgpath , save_path)
                if bbox_list and outimgnum % 10 == 0:
                    shutil.copyfile(imgpath,save_path)
                    annot_dict = [('input', save_path), ('target', bbox_list)]
                    dict1 = dict(annot_dict)
                    json_list.append(dict1)

        with open(jsonpath, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def tp_list(preds, targets):
        tplist = []
        tp, fp = 0, 0
        iou_th = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for _iou in iou_th:
            for predbox in preds:
                rats = []
                for tarbox in targets:
                    rat = iou_xywh(predbox[2:6], tarbox[1:5])
                    rats.append(rat)
                # print(max(rats))
                if max(rats) >= _iou:
                    tp += 1
                else:
                    fp += 1
            tplist.append(tp)
        return tplist

def ccpd_json_fromtxt(txtpath, imgpath, label_path, log=''):
        file = open(txtpath)
        json_list = []
        while 1:
            lines = file.readlines(240000)
            if not lines:
                break
            for line in lines:
                imgname = imgpath + '/' + line
                imgname = imgname.replace('\n', '')
                target_list = []
                # print('start',imgname,'end')
                if os.path.isfile(imgname):
                    targets = [0]
                    errorup = imgname
                    name = line.split('/')[-1]
                    target = name.split('-')[3]
                    point1, point2, point3, point4 = target.split('_')[0].split('&'), target.split('_')[1].split('&'), \
                                                     target.split('_')[2].split('&'), target.split('_')[3].split('&')
                    # print(point1,point2,point3,point4)
                    pointmax = [max(int(point1[0]), int(point4[0])), max(int(point1[1]), int(point2[1]))]
                    pointmin = [min(int(point3[0]), int(point2[0])), min(int(point3[1]), int(point4[1]))]
                    pointcenter = [int((pointmin[0] + pointmax[0]) / 2), int((pointmin[1] + pointmax[1]) / 2)]
                    w, h = int(pointmax[0] - pointmin[0]), int(pointmax[1] - pointmin[1])
                    targets.append(pointcenter[0])
                    targets.append(pointcenter[1])
                    targets.append(w)
                    targets.append(h)
                    target_list.append(targets)
                    # print(pointcenter,w,h)
                    annot_dict = [('input', imgname), ('target', target_list)]
                    dict1 = dict(annot_dict)
                    print(annot_dict)
                    json_list.append(dict1)
            with open(label_path, 'w', encoding='utf-8') as f:
                for sample in json_list:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        file.close()
        return 'ok'

def ccpd_square(jsonpath, newpath, labelpath):
        json_list = []
        f = open(jsonpath)
        lines = f.readlines()
        print(len(lines))
        for i, line in enumerate(lines):
            info = json.loads(line)
            if os.path.isfile(info['input']):
                # print(info['input'])
                image = cv2.imread(info['input'])
                height, width, _ = image.shape
                if height > width:

                    lenght = height - width
                    print('height', lenght)
                    a = cv2.copyMakeBorder(image, 0, 0, 0, lenght, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                elif width > height:
                    lenght = height - width
                    a = cv2.copyMakeBorder(image, 0, lenght, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # image_show('a',a)
                filename = info['input'].split('/')[-1]
                savepath = os.path.join(newpath, filename)
                # cv2.imwrite(savepath,a)
                annot_dict = [('input', savepath), ('target', info['target'])]
                dict1 = dict(annot_dict)
                print(i)
                json_list.append(dict1)

        with open(labelpath, 'w', encoding='utf-8') as f:
            print(len(json_list))
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def cutjson_ccpdtest(jsonpath, out):
        filelist = [[], [], [], [], [], [], [], [], []]
        bboxlist = []
        errorlist = []
        words = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_np', 'ccpd_rotate',
                 'ccpd_tilt', 'ccpd_weather']
        for word in words:
            os.mknod(out + '/' + f"test_{word}.json")
        f = open(jsonpath)
        lines = f.readlines()
        for line in lines:
            boxnum = 0
            info = json.loads(line)
            if os.path.isfile(info['input']):
                for i, word in enumerate(words):
                    if word in info['input']:
                        filelist[i].append(info)
        for x, ls in enumerate(filelist):
            with open(out + '/' + f'tese_{words[x]}.json', 'w', encoding='utf-8') as newf:
                for sample in ls:
                    newf.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ccpd_checkjson(jsonpath):
        f = open(jsonpath)
        lines = f.readlines()
        os.makedirs("ccpdcheck/", exist_ok=True)
        for i, line in enumerate(lines):
            boxnum = 0
            info = json.loads(line)
            image = cv2.imread(info['input'])
            file = info['input'].split('/')[-1]
            for target in info['target']:
                target = xywh2xyxy(target[1:], 0)
                cv2.rectangle(image, (target[1], target[2]), (target[3], target[4]), (0, 0, 255), 1)
            cv2.imwrite('ccpdcheck/' + file, image)

def ccpd_fromxml(imgpath, xmlpath, labelpath):
        '''

        :param imgpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/"
        :param xmlpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/annot/"
        :param jsonpath: "/home/linwang/pyproject/TrainSet/detection/v32/label_fix.json"
        :return:
        '''
        words = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_np', 'ccpd_rotate',
                 'ccpd_tilt',
                 'ccpd_weather']
        json_list = []
        for word in words:
            _imgpath = imgpath + word
            imgs = [os.path.join(_imgpath, s) for s in os.listdir(_imgpath)]  # 训练样本的img路径
            for img in imgs:
                filename = img.split('/')[-1]
                x = filename.split('.')[-1]
                filename_img = filename.replace(f'.{x}', '')
                file = img.split('/')[-2] + '/' + img.split('/')[-1]
                # filename_img = filename.split('.')[-2]
                annot = xmlpath + filename_img + '.xml'
                et = ET.parse(annot)
                element = et.getroot()
                element_objs = element.findall('object')
                target_list = []
                for element_obj in element_objs:
                    reframe = []
                    # class_name = element_obj.find('name').text
                    # if class_name == '小车':
                    #    first = 0
                    # else:
                    #    first = 1
                    reframe.append(int(element_obj.find('bndbox').find('xmin').text))
                    reframe.append(int(element_obj.find('bndbox').find('ymin').text))
                    reframe.append(int(element_obj.find('bndbox').find('xmax').text))
                    reframe.append(int(element_obj.find('bndbox').find('ymax').text))
                    trframe = xyxy2xywh_c(reframe, 0)
                    target_list.append(trframe)
                annot_dict = [('input', file), ('target', target_list)]
                dict1 = dict(annot_dict)
                json_list.append(dict1)
        with open(labelpath, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ccpd_fixjson(fixjson, jsonpath):
        '''

        :param fixjson: "/home/linwang/pyproject/TrainSet/detection/v32/label_fix.json"
        : jsonpath /home/linwang/pyproject/TrainSet/detection/v32/
        :return:
        '''
        os.makedirs(jsonpath + 'test', exist_ok=True)
        words = ['train', 'val', 'test']
        # for word in words:
        # s    os.mknod(jsonpath + '/' + f"test_{word}.json")
        f = open(fixjson, 'r')
        lines = f.readlines()
        for word in words:
            json_list = []
            _jsonpath = jsonpath + '/' + f'label_{word}.json'
            f_old = open(_jsonpath)
            lines_old = f_old.readlines()
            for line_old in tqdm.tqdm(lines_old):
                num = 0
                info_old = json.loads(line_old)
                file_old = info_old['input']
                filename_old = file_old.split('/')[-2] + '/' + file_old.split('/')[-1]
                for line in lines:
                    info = json.loads(line)
                    file = info['input']
                    filename = file.split('/')[-2] + '/' + file.split('/')[-1]
                    if filename == filename_old:
                        annot_dict = [('input', file_old), ('target', info['target'])]
                        dict1 = dict(annot_dict)
                        json_list.append(dict1)
                        num = 1
                        img = cv2.imread(file_old)
                        for box in info['target']:
                            box1 = xywh2xyxy(box[1:5], 0)
                            cv2.rectangle(img, (box1[1], box1[2]), (box1[3], box1[4]), (255, 0, 0), 2)
                        cv2.imwrite(jsonpath + 'test' + '/' + file_old.split('/')[-1], img)
                if num == 0:
                    json_list.append(info_old)
            labelpath = jsonpath + '/' + f"test_{word}.json"
            with open(labelpath, 'w', encoding='utf-8') as f:
                for sample in json_list:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ccpd_fromxml_addyellow(imgpath, xmlpath, labelpath):
        '''

        :param imgpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/"
        :param xmlpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/annot/"
        :param jsonpath: "/home/linwang/pyproject/TrainSet/detection/v32/label_fix.json"
        :return:
        '''
        annots = []
        json_list = []
        for s in os.listdir(xmlpath):
            if s[-3:] == 'xml':
                annots.append(os.path.join(xmlpath, s))  # 训练样本的xml路径
        for annot in annots:
            """依次解析XML文件"""
            filename = annot.split('/')[-1]
            x = filename.split('.')[-1]
            filename_img = filename.replace(f'.{x}', '')
            # filename_img = filename.split('.')[-2]
            file_path = imgpath + '/' + filename_img + '.jpg'
            print(file_path)
            if os.path.isfile(file_path):
                et = ET.parse(annot)
                element = et.getroot()
                element_objs = element.findall('object')
                target_list = []
                for element_obj in element_objs:
                    reframe = []
                    reframe.append(int(element_obj.find('bndbox').find('xmin').text))
                    reframe.append(int(element_obj.find('bndbox').find('ymin').text))
                    reframe.append(int(element_obj.find('bndbox').find('xmax').text))
                    reframe.append(int(element_obj.find('bndbox').find('ymax').text))
                    trframe = xyxy2xywh_c(reframe)
                    target_list.append(trframe)
                annot_dict = [('input', file_path), ('target', target_list)]
                dict1 = dict(annot_dict)
                json_list.append(dict1)
        with open(labelpath, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ccpd_addyel_jsoncut(jsonpath, out_blue, out_yel, out_gre, out_wh, out_other):
        f = open(jsonpath, 'r+', encoding='UTF-8')
        lines = f.readlines()
        jsonlist_blue = []
        jsonlist_yel = []
        jsonlist_gre = []
        jsonlist_wh = []
        jsonlist_other = []
        for line in lines:
            info = json.loads(line)
            x = info['input']
            # print(x,x[-3:])
            if x[-3:] == 'jpg':
                x = x.replace('.jpg', '')
                x = x.split('_')[-1]
                if x == '0':
                    jsonlist_blue.append(info)
                elif x == '1':
                    jsonlist_yel.append(info)
                elif x == '2':
                    jsonlist_gre.append(info)
                elif x == '4':
                    jsonlist_wh.append(info)
                else:
                    jsonlist_other.append(info)
        with open(out_blue, 'w', encoding='utf-8') as newf:
            for sample in jsonlist_blue:
                newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(out_yel, 'w', encoding='utf-8') as newf:
            for sample in jsonlist_yel:
                newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(out_gre, 'w', encoding='utf-8') as newf:
            for sample in jsonlist_gre:
                newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(out_wh, 'w', encoding='utf-8') as newf:
            for sample in jsonlist_wh:
                newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with open(out_other, 'w', encoding='utf-8') as newf:
            for sample in jsonlist_other:
                newf.write(json.dumps(sample, ensure_ascii=False) + "\n")

def xyxy2xywh_c(frame):
    trframe = []
    trframe.append(int((frame[0] + frame[2]) / 2))
    trframe.append(int((frame[1] + frame[3]) / 2))
    trframe.append(int(frame[2] - frame[0]))
    trframe.append(int(frame[3] - frame[1]))
    # print(frame,trframe)
    return trframe

def xywh2xywh_c(box):
    newbox = [0,0,0,0]
    newbox[0] = box[0] + int(box[2]/2)
    newbox[1] = box[1] + int(box[3] / 2)
    newbox[2] = box [2]
    newbox[3] = box [3]
    return newbox
def xywh_c2xyxy(frame):
    trframe = []
    trframe.append(frame[0] - (frame[2] / 2))
    trframe.append(frame[1] - (frame[3] / 2))
    trframe.append(frame[0] + (frame[2] / 2))
    trframe.append(frame[1] + (frame[3] / 2))
    return trframe
def xywh2xyxy(frame):
    trframe = []
    trframe.append(int(frame[0]))
    trframe.append(int(frame[1]))
    trframe.append(int(frame[0] + (frame[2])))
    trframe.append(int(frame[1] + (frame[3])))
    return trframe
def iou_xy(Reframe,GTframe):
    #print('正在解析 annotation files')
    # 得到第一个矩形的左上坐标及宽和高
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    # 得到第二个矩形的左上坐标及宽和高
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    # 计算重叠部分的宽和高
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    # 如果重叠部分为负, 即不重叠
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)

    return ratio

def iou_xywh(Reframe,GTframe):
    # print('正在解析 annotation files')
    # 得到第一个矩形的左上坐标及宽和高
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    # 得到第二个矩形的左上坐标及宽和高
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]

    # 计算重叠部分的宽和高
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    # 如果重叠部分为负, 即不重叠
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio

def clear_box_pred(pred,targets):
    pre_move = []
    tar_move = []
    c1,c2= [],[]
    for i, sample in enumerate(pred):
        if (sample[4]*sample[5]) < 2500 or sample[4]<50 or sample[5]<40:
            pre_move.append(i)
    for j, sample1 in enumerate(targets):
        if (sample1[3] * sample1[4]) < 2500 or sample1[3]<50 or sample1[4]<40:
            tar_move.append(j)
    c1 = c1+pred
    for i in pre_move:
        c1.remove(pred[i])
    c2 = c2 + targets
    for j in tar_move:
        c2.remove(targets[j])
    #print('pred', c1,'\n','tar',c2)
    return c1,c2

def padding(image):
    height, width, _ = image.shape
    if height > width:
        lenght = height - width
        image = cv2.copyMakeBorder(image, 0, 0, 0, lenght, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif width > height:
        lenght = width - height
        image = cv2.copyMakeBorder(image, 0, lenght, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image

def gtout(jsondir,savedir):
    f = open(jsondir, 'r+', encoding='UTF-8')
    lines = f.readlines()
    for line in lines:
            info = json.loads(line)
            if os.path.isfile(info['input']):
                imgname = info['input'].split('/')[-1]
                img = cv2.imread(info['input'])
                for sample in info['target']:
                    # import pdb
                    # pdb.set_trace()
                    cv2.rectangle(img,(sample[0],sample[1]),(sample[2],sample[3]),(255,0,0),3)
                cv2.imwrite(savedir + imgname,img)
def walkfile(path):
    filelist = []
    # import pdb
    # pdb.set_trace()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            filelist.append(os.path.join(root, name))
        # for name in dirs:
        #     pass
    return filelist
def ourdata(imgspath,labelspath):
    imgs_dir = [os.path.join(imgspath, f) for f in
                os.listdir(imgspath) if f.endswith('jpg')]
    labels_dir = [os.path.join(labelspath, f) for f in
                os.listdir(labelspath) if f.endswith('txt')]
    import pdb
    pdb.set_trace()
    if True:
        os.makedirs("/data/linwang/anomaiy/images/test/", exist_ok=True)
        os.makedirs("/data/linwang/anomaiy/labels/test/", exist_ok=True)
        os.makedirs("/data/linwang/anomaiy/images/trains/", exist_ok=True)
        os.makedirs("/data/linwang/anomaiy/labels/trains/", exist_ok=True)
        os.makedirs("/data/linwang/anomaiy/images/valids/", exist_ok=True)
        os.makedirs("/data/linwang/anomaiy/labels/valids/", exist_ok=True)
        count = 0
        for labelpath  in tqdm.tqdm(labels_dir):
            if count < 1000:
                tem = labelpath.replace('savetxt', 'JPEGImages')
                img = tem.replace('txt', 'jpg')
                if os.path.isfile(img):
                    shutil.copy(labelpath,"/data/linwang/anomaiy/labels/test/")
                    shutil.copy(img,"/data/linwang/anomaiy/images/test/")
            if count >= 1000 and count < 6000:

                tem = labelpath.replace('savetxt', 'JPEGImages')
                img = tem.replace('txt', 'jpg')
                if os.path.isfile(img):
                    shutil.copy(labelpath, "/data/linwang/anomaiy/labels/valids/")
                    shutil.copy(img, "/data/linwang/anomaiy/images/valids/")
            if count >= 6000:

                tem = labelpath.replace('savetxt', 'JPEGImages')
                img = tem.replace('txt', 'jpg')
                if os.path.isfile(img):
                    shutil.copy(labelpath, "/data/linwang/anomaiy/labels/trains/")
                    shutil.copy(img, "/data/linwang/anomaiy/images/trains/")
            count += 1
    return 0
def txt2json(input,jsonpath):

    imgs_dir = [x for x in walkfile(input) if x.endswith('jpg')]
    labels_dir = [x for x in walkfile(input) if x.endswith('txt')]
    print(len(imgs_dir),len(labels_dir))
    # import pdb
    # pdb.set_trace()
    json_list = []
    savenum = 0
    for label in tqdm.tqdm(labels_dir):
        target_list = []
        tem = label.replace('labels','images')
        imgpath = tem.replace('txt','jpg')
        if os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            h,w,_ = img.shape
            with open(label, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            if len(lb) != 0:
                lb[:,1:] *= [w, h, w, h]
            else:
                continue
            for j, x in enumerate(lb):
                c = int(x[0])  # class
                b = xywh_c2xyxy(x[1:])
                b.append(c)
                target_list.append(b)
            if savenum < 20:
                for b in target_list:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
                cv2.imwrite(imgpath.replace('images','samples'),img)
                savenum += 1
            if target_list is not []:
                annot_dict = [('input', imgpath), ('target', target_list)]
                dict1 = dict(annot_dict)
                json_list.append(dict1)
    with open(jsonpath, 'w', encoding='utf-8') as f:
        for sample in json_list:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            # cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(255,0,0),2)
        # cv2.imwrite('debug.jpg',img)
def wider2yolo5(labelspath,outpath):
    import pdb
    pdb.set_trace()
    labels = glob.glob(labelspath + '*.txt')
    os.makedirs(outpath + 'images/', exist_ok=True)
    os.makedirs(outpath + 'labels/', exist_ok=True)
    for samples in tqdm.tqdm(labels):
        name = (samples.split('/')[-1]).replace('.txt','').replace('.jpg','')
        tem = samples.replace("Annotations/",'Images/')
        imgname = tem.replace('.txt','')
        f = open(samples,'r')
        l = [x.split() for x in f.read().strip().splitlines()]
        if os.path.isfile(imgname):
            img = cv2.imread(imgname)
            h,w,_ = img.shape
            newlabel = open(outpath + 'labels/' + 'wider' + name + '.txt','w')
            for j, sample in enumerate(l):
                if j > 0:
                    cla , x1,y1,x2,y2 = [int(x) for x in sample ]
                    if cla in [0,1,2,3,4,5]:
                        newbox = xyxy2xywh_c([x1,y1,x2,y2])
                        newbox[0] = newbox[0]/w
                        newbox[1] = newbox[1]/h
                        newbox[2] = newbox[2]/w
                        newbox[3] = newbox[3]/h
                        newlabel.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
            cv2.imwrite(outpath + 'images/' + 'wider' + name + '.jpg' , img)
    return
def meva2yolo5(labelspath,videospath,outspath):
    labels = [file for file in walkfile(labelspath) if file.endswith('yml')]
    videos = [file for file in walkfile(videospath) if file.endswith('avi')]
    outimg,outlabel = outspath,outspath.replace('images','labels')
    for video_path in tqdm.tqdm(videos):
        videoname = video_path.split(os.sep)[-1].replace('.r13.avi','')
        outdir = os.path.join(outlabel,video_path.split(os.sep)[-3],video_path.split(os.sep)[-2],videoname)
        os.makedirs(outdir,exist_ok=True)
        label_geom_path = video_path.replace(videospath,labelspath).replace('r13.avi','geom.yml')
        label_type_path = video_path.replace(videospath, labelspath).replace('r13.avi', 'types.yml')
        if os.path.isfile(label_geom_path) and os.path.isfile(label_type_path):
            typedict = {}
            lb = []    #x,y,x,y,class,framenum
            label_geom = yaml.load(open(label_geom_path,'r'),Loader=yaml.FullLoader)
            label_type = yaml.load(open(label_type_path,'r'),Loader=yaml.FullLoader)
            if len(label_geom) > 100:
                for _type in label_type:
                    typedict[_type['types']['id1']] = 0 if list(_type['types']['cset3'].keys())[0] == 'person' else 1
                for _geom in label_geom:
                    _label = _geom['geom']['g0'].split()
                    _label.append(typedict[_geom['geom']['id1']])
                    _label.append(_geom['geom']['ts0'])
                    lb.append(_label)
                lb = np.array(lb)

                for sample in lb:
                    with open(outdir + os.sep+videoname+'_'+sample[5]+'.txt','a') as newf: #not resize for w,h
                        newf.write(sample[4] + " " + " ".join([str(a) for a in sample[0:4]]) + '\n')

            # pdb.set_trace()


def pennfudanped2yolo5(inputpath,outpath):

    pngpath = inputpath + 'PNGImages/'
    oritxt = inputpath + 'Annotation/'
    newtxt = outpath + 'labels/'
    savepath = outpath + 'images/'
    os.makedirs(newtxt, exist_ok=True)
    os.makedirs(savepath, exist_ok=True)
    txtfiles = os.listdir(oritxt)
    matrixs = []
    index = 1
    for txtpath in txtfiles:
        img = cv2.imread(pngpath + txtpath[:-4] + ".png")
        h,w,_ = img.shape
        f1 = open(oritxt + txtpath, 'r')
        newf = open(newtxt + txtpath,'w')
        for line in f1.readlines():
            if re.findall('Xmin', line):
                pt = [int(x) for x in re.findall(r"\d+", line)]
                matrixs.append(pt)
                # cv2.rectangle(img, (pt[1], pt[2]), (pt[3], pt[4]), (0, 255, 0), 2)
                # tmp = img[pt[2]:pt[4], pt[1]:pt[3]]
                if True:
                    newbox = xyxy2xywh_c(pt[1:])
                    newbox[0] = newbox[0] / w
                    newbox[1] = newbox[1] / h
                    newbox[2] = newbox[2] / w
                    newbox[3] = newbox[3] / h
                    newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
        cv2.imwrite(savepath +   'penn' + txtpath[:-4] + '.jpg', img)
def coco2yolo5(inputpath,inputlabel,outpath):
    """
    cityperson(coco format) > yolo5
    """
    f = open(inputlabel, 'r+', encoding='UTF-8')
    lines = f.readlines()
    pngpath = inputpath
    newtxt = outpath + 'labels/'
    savepath = outpath + 'images/'
    os.makedirs(newtxt, exist_ok=True)
    os.makedirs(savepath, exist_ok=True)
    for line in lines:
        info = json.loads(line)
        for image in tqdm.tqdm(info['images']):

                inputimgpath = pngpath + image['file_name']
                txtpath = newtxt + image['file_name'].replace('png', 'txt')
                outimagepath = savepath + image['file_name'].replace('png', 'jpg')
                os.makedirs(os.path.dirname(txtpath), exist_ok=True)
                os.makedirs(os.path.dirname(outimagepath), exist_ok=True)
                newf = open(txtpath, 'w+')
                if os.path.isfile(inputimgpath):
                    for sample in info['annotations']:
                        imageid = sample['image_id']
                        categoryid = sample['category_id']
                        box = [int(x) for x in sample['bbox']]
                        if int(image['id']) == int(imageid) and categoryid ==1 : # note
                            w = image['width']
                            h = image['height']
                            newbox = xywh2xywh_c(box)
                            newbox[0] = newbox[0] / w
                            newbox[1] = newbox[1] / h
                            newbox[2] = newbox[2] / w
                            newbox[3] = newbox[3] / h
                            newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n') # note
                shutil.copyfile(inputimgpath,outimagepath)
def mot2yolo5(inputpath, outpath):
    dirlists = os.listdir(inputpath)
    for dirlist in dirlists:
        gtpath = inputpath + dirlist + '/gt/gt.txt'
        imgs = inputpath + dirlist + '/img1/'
        outimgs = outpath + 'images/'
        outlabel = outpath + 'labels/'
        os.makedirs(os.path.dirname(outimgs), exist_ok=True)
        os.makedirs(os.path.dirname(outlabel), exist_ok=True)
        # setfile = open(inputpath + dirlist + '/seqinfo.ini')
        setfile = configparser.ConfigParser()
        setfile.read(inputpath + dirlist + '/seqinfo.ini', encoding="utf-8")  # python3
        sections = setfile.sections()
        w = int(setfile.items('Sequence')[4][1])
        h = int(setfile.items('Sequence')[5][1])
        f = open(gtpath,'r')
        lb = np.array([x.split(',') for x in f.read().strip().splitlines()], dtype=np.float16)  # labels
        filelist = []
        for j, sample in tqdm.tqdm(enumerate(lb),ncols =  len(lb)):
            if sample[7] == 7 or sample [7] == 1:
                imgname = str(int(sample[0])).zfill(6)
                imgpath = imgs + imgname + '.jpg'
                labelpath = outlabel + dirlist + '_' +imgname + '.txt'
                outimgpath = outimgs + dirlist + '_' +imgname + '.jpg'
                box = [int(x) for x in sample[2:6]]
                # import pdb
                # pdb.set_trace()
                newbox = xywh2xywh_c(box)
                newbox[0] = newbox[0] / w
                newbox[1] = newbox[1] / h
                newbox[2] = newbox[2] / w
                newbox[3] = newbox[3] / h
                if os.path.isfile(imgpath):
                    newf = open(labelpath, 'a+')
                    newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')  # note
                    newf.close()
                    if imgpath not in filelist:
                        shutil.copyfile(imgpath, outimgpath)
                        filelist.append(imgpath)
def cuhk_square2yolo5(inputpath, outpath):
    gtfile = inputpath + 'train_ground_truth_data.mat'
    import h5py
    feature=h5py.File(gtfile)
    imglist = feature['ground_truth_data']['list']
    # data = feature['feature_data'][:]

    import pdb
    pdb.set_trace()
def central2yolo5(inputpath,outpath):
    imgspath = inputpath + 'image/'
    gtpath = inputpath + 'annotations/'
    outimgpath = outpath + 'images/'
    labelpath = outpath + 'labels/'
    os.makedirs(os.path.dirname(outimgpath), exist_ok=True)
    os.makedirs(os.path.dirname(labelpath), exist_ok=True)
    for gt in glob.glob(gtpath + '*.txt'):
        h = 480
        w = 640
        f = open(gt,'r')
        import  pdb
        pdb.set_trace()
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            imgname = line.split(':')[0].replace('"','').split('/')[-1]
            imgpath = input + 'image/' + line.split(':')[0].replace('"','')
            labels = re.findall(r"\d+", line.split(':')[1])
            newlabel = labelpath + imgname.replace('jpg','txt')
            saveimg = outimgpath + imgname
            newf = open(newlabel,'w')
            box = []
            if os.path.isfile(imgpath):
                for pixel in labels:
                    box.append(int(pixel))
                    if len(box) == 4:
                        newbox = xyxy2xywh_c(box)
                        newbox[0] = newbox[0] / w
                        newbox[1] = newbox[1] / h
                        newbox[2] = newbox[2] / w
                        newbox[3] = newbox[3] / h
                        newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
                        box = []
                shutil.copyfile(imgpath, saveimg)

    return
def vbb2dict(input, output, person_types=None):
    """
    Parse caltech vbb annotation file to dict
    Args:
        vbb_file: input vbb file path
        cam_id: camera id
        person_types: list of person type that will be used (total 4 types: person, person-fa, person?, people).
            If None, all will be used:
    Return:
        Annotation info dict with filename as key and anno info as value
    """
    seqspath = input + 'seqs/'
    gtpath = input + 'annotations/'
    outimgpath = output + 'images/'
    labelpath = output + 'labels/'
    os.makedirs(os.path.dirname(outimgpath), exist_ok=True)
    os.makedirs(os.path.dirname(labelpath), exist_ok=True)
    for dirs in os.listdir(gtpath):
        for vbbname in os.listdir(gtpath + dirs):
            vbbpath = gtpath + dirs + '/' + vbbname
            seq_file = seqspath + dirs + '/' + vbbname.replace('vbb','seq')
            annos = defaultdict(dict)
            vbb = scio.loadmat(vbbpath)
            # object info in each frame: id, pos, occlusion, lock, posv
            objLists = vbb['A'][0][0][1][0]
            objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
            # person index
            if not person_types:
                person_types = ["person", "person-fa", "person?", "people"]
            person_index_list = [x for x in range(len(objLbl)) if objLbl[x] in person_types]
            for frame_id, obj in enumerate(objLists):
                if len(obj) > 0:
                    frame_name = str(dirs) + '_' + str(vbbname.replace('.vbb','')) + '_' + str(frame_id+1) + ".jpg"
                    annos[frame_name] = defaultdict(list)
                    annos[frame_name]["id"] = frame_name
                    for fid, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                        fid = int(fid[0][0]) - 1  # for matlab start from 1 not 0
                        if not fid in person_index_list:  # only use bbox whose label is given person type
                            continue
                        annos[frame_name]["label"] = objLbl[fid]
                        pos = pos[0].tolist()
                        occl = int(occl[0][0])
                        annos[frame_name]["occlusion"].append(occl)
                        annos[frame_name]["bbox"].append(pos)
                    if not annos[frame_name]["bbox"]:
                        del annos[frame_name]
            cap = cv2.VideoCapture(seq_file)
            index = 1
            # captured frame list
            v_id = os.path.splitext(os.path.basename(seq_file))[0]
            cap_frames_index = np.sort([int(os.path.splitext(id)[0].split("_")[2]) for id in annos.keys()])
            while True:
                ret, frame = cap.read()
                if ret:
                    if not index in cap_frames_index:
                        index += 1
                        continue
                    # import pdb
                    # pdb.set_trace()
                    outname = os.path.join(outimgpath, str(dirs) + '_' + str(vbbname.replace('.vbb','')) + '_' + str(index) + ".jpg")
                    print("Current frame: ", v_id, str(index))
                    cv2.imwrite(outname, frame)
                    h, w, _ = frame.shape
                    newf = open(labelpath + str(dirs) + '_' + str(vbbname.replace('.vbb','')) + '_' + str(index) + '.txt','w')
                    for box in annos[str(dirs) + '_' + str(vbbname.replace('.vbb','')) + '_' + str(index) + ".jpg"]['bbox']:
                        newbox = xywh2xywh_c([int(x) for x in box])
                        newbox[0] = newbox[0] / w
                        newbox[1] = newbox[1] / h
                        newbox[2] = newbox[2] / w
                        newbox[3] = newbox[3] / h
                        newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
                    newf.close()
                else:
                    break
                index += 1

    return
def vis_yolo5(inputpath):
    files = walkfile(inputpath)
    txtfiles = [x for x in files if x.endswith('txt')]
    os.makedirs('/data/linwang/vis_debug/',exist_ok=True)
    bug = open('/data/linwang/debug.txt', 'w')
    for file in tqdm.tqdm(txtfiles):
        tem  = file.replace('txt','jpg')
        imgpath = tem.replace('labels','images')
        if os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            imgname = imgpath.split('/')[-1]
            h, w, _ = img.shape
            with open(file, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

            for j, x in enumerate(lb):
                c = int(x[0])  # class
                b = x[1:] * [w, h, w, h]  # box
                b = xywh_c2xyxy(b)
                cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(255,0,0),2)
            cv2.imwrite('/data/linwang/vis_debug/'+ imgname,img)
        else:
            bug.write(file + '\n')
    bug.close()

def bddperson(input,output):
    inputimgpath = input + 'images/'
    inputlabel = input + 'labels/valids/'
    outimgpath = output + 'images/'
    labelpath = output + 'labels/'
    os.makedirs(os.path.dirname(outimgpath), exist_ok=True)
    os.makedirs(os.path.dirname(labelpath), exist_ok=True)
    for label in tqdm.tqdm(glob.glob(inputlabel + '*.txt')):
        # import pdb
        # pdb.set_trace()
        inputimg = label.replace('labels','images').replace('txt','jpg')
        if os.path.isfile(inputimg):
            mark = 0
            newlabel = labelpath + label.split(os.sep)[-1]
            newimg = outimgpath + label.split(os.sep)[-1].replace('txt','jpg')
            with open(label, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            for j, x in enumerate(lb):
                c = int(x[0])  # class
                b = x[1:]
                if c == 0:
                    mark = 1
                    newf = open(newlabel, 'a+')
                    newf.write('0' + " " + " ".join([str(a) for a in b]) + '\n')
            if mark == 1:
                shutil.copyfile(inputimg,newimg)
    return
def bddveh(input,output):
    inputimgpath = input + 'images/'
    inputlabel = input + 'labels/trains/'
    outimgpath = output + 'images/'
    labelpath = output + 'labels/'
    os.makedirs(os.path.dirname(outimgpath), exist_ok=True)
    os.makedirs(os.path.dirname(labelpath), exist_ok=True)
    for label in tqdm.tqdm(glob.glob(inputlabel + '*.txt')):
        # import pdb
        # pdb.set_trace()
        inputimg = label.replace('labels','images').replace('txt','jpg')
        if os.path.isfile(inputimg):
            mark = 0
            newlabel = labelpath + label.split(os.sep)[-1]
            newimg = outimgpath + label.split(os.sep)[-1].replace('txt','jpg')
            with open(label, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            for j, x in enumerate(lb):
                c = int(x[0])  # class
                b = x[1:]
                if c in [2,3,4]:
                    mark = 1
                    newf = open(newlabel, 'a+')
                    newf.write('1' + " " + " ".join([str(a) for a in b]) + '\n')
            if mark == 1:
                shutil.copyfile(inputimg,newimg)
    return
def cocoperson(input,output):
    inputimgpath = input + 'images/'
    inputlabel = input + 'labels/train2017/'
    outimgpath = output + 'images/'
    labelpath = output + 'labels/'
    os.makedirs(os.path.dirname(outimgpath), exist_ok=True)
    os.makedirs(os.path.dirname(labelpath), exist_ok=True)
    for label in tqdm.tqdm(glob.glob(inputlabel + '*.txt')):
        # import pdb
        # pdb.set_trace()
        inputimg = label.replace('labels','images').replace('txt','jpg')
        if os.path.isfile(inputimg):
            mark = 0
            newlabel = labelpath + label.split(os.sep)[-1]
            newimg = outimgpath + label.split(os.sep)[-1].replace('txt','jpg')
            with open(label, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            for j, x in enumerate(lb):
                c = int(x[0])  # class
                b = x[1:]
                if c == 0:
                    mark = 1
                    newf = open(newlabel, 'a+')
                    newf.write('0' + " " + " ".join([str(a) for a in b]) + '\n')
            if mark == 1:
                shutil.copyfile(inputimg,newimg)
    return
def cocoveh(input,output):
    inputimgpath = input + 'images/'
    inputlabel = input + 'labels/train2017/'
    outimgpath = output + 'images/'
    labelpath = output + 'labels/'
    os.makedirs(os.path.dirname(outimgpath), exist_ok=True)
    os.makedirs(os.path.dirname(labelpath), exist_ok=True)
    for label in tqdm.tqdm(glob.glob(inputlabel + '*.txt')):
        # import pdb
        # pdb.set_trace()
        inputimg = label.replace('labels','images').replace('txt','jpg')
        if os.path.isfile(inputimg):
            mark = 0
            newlabel = labelpath + label.split(os.sep)[-1]
            newimg = outimgpath + label.split(os.sep)[-1].replace('txt','jpg')
            with open(label, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            for j, x in enumerate(lb):
                c = int(x[0])  # class
                b = x[1:]
                if c in [2,3,5,6,7]:
                    mark = 1
                    newf = open(newlabel, 'a+')
                    newf.write('1' + " " + " ".join([str(a) for a in b]) + '\n')
            if mark == 1:
                shutil.copyfile(inputimg,newimg)
    return

def all_ped(inputlist):
    outimgs_train = "/data/linwang/all_ped/images/trains/"
    outimgs_val = "/data/linwang/all_ped/images/valids/"
    outlabels_train = "/data/linwang/all_ped/labels/trains/"
    outlabels_val = "/data/linwang/all_ped/labels/valids/"
    os.makedirs(outimgs_train,exist_ok=True)
    os.makedirs(outimgs_val, exist_ok=True)
    os.makedirs(outlabels_train, exist_ok=True)
    os.makedirs(outlabels_val, exist_ok=True)
    for onepath in inputlist:
        # import pdb
        # pdb.set_trace()
        labels = onepath + 'labels/'
        imgs = onepath + 'images/'
        index = 0
        for label in tqdm.tqdm([x for x in walkfile(onepath) if x.endswith('txt')]):
            inputimg = label.replace('labels','images').replace('txt','jpg')
            if os.path.isfile(inputimg):
                index += 1
                if index%15 == 0:
                    shutil.copy(inputimg,outimgs_val)
                    shutil.copy(label,outlabels_val)
                else:
                    shutil.copy(inputimg, outimgs_train)
                    shutil.copy(label, outlabels_train)
    return
def all_veh(inputlist):
    outimgs_train = "/data/linwang/all_veh/images/trains/"
    outimgs_val = "/data/linwang/all_veh/images/valids/"
    outlabels_train = "/data/linwang/all_veh/labels/trains/"
    outlabels_val = "/data/linwang/all_veh/labels/valids/"
    os.makedirs(outimgs_train,exist_ok=True)
    os.makedirs(outimgs_val, exist_ok=True)
    os.makedirs(outlabels_train, exist_ok=True)
    os.makedirs(outlabels_val, exist_ok=True)
    for onepath in inputlist:
        # import pdb
        # pdb.set_trace()
        labels = onepath + 'labels/'
        imgs = onepath + 'images/'
        index = 0
        for label in tqdm.tqdm(glob.glob(labels + '*.txt')):
            inputimg = label.replace('labels','images').replace('txt','jpg')
            if os.path.isfile(inputimg):
                index += 1
                if index%20 == 0:
                    shutil.copy(inputimg,outimgs_val)
                    shutil.copy(label,outlabels_val)
                else:
                    shutil.copy(inputimg, outimgs_train)
                    shutil.copy(label, outlabels_train)
    return
def uadet2yolo5(inpath,imgdir,outpath):
    '''

            :param inpath: xml
            :imgpath :img
            :param outpath:  txt or xml or json savepath
            :return:
            '''
    annots = [os.path.join(inpath, s) for s in os.listdir(inpath)]  # 训练样本的xml路径
    outimgnum = 0
    json_list = []
    json_list_rainy = []
    json_list_night = []
    json_list_cloudy = []
    out_img_path = outpath +  'images/'
    outlabel = outpath + 'labels/'
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(outlabel, exist_ok=True)
    for annot in tqdm.tqdm(annots):
        """依次解析XML文件"""
        et = ET.parse(annot)
        element = et.getroot()
        filename = annot.split('/')[-1]
        imgdirname = filename.replace('.xml', '')
        # element_objs = element.findall('target')
        # tag = element.tag
        # attrib = element.attrib['name']  # zheshi tupianwenjianjia he xml mingzi
        # value = element.text
        idx = 1
        for child in element:
            bbox_list = []
            # print('1', child.attrib)
            for sec_child in child:
                # print('2', sec_child.attrib)
                for tri_child in sec_child:
                    # print('3' ,tri_child.tag, tri_child.attrib)
                    for four_child in tri_child:
                        if four_child.tag == 'box':
                            bbox = []
                            bbox.append(
                                int(float(four_child.attrib['left']) + float(four_child.attrib['width']) / 2))
                            bbox.append(
                                int(float(four_child.attrib['top']) + float(four_child.attrib['height']) / 2))
                            bbox.append(int(float(four_child.attrib['width'])))
                            bbox.append(int(float(four_child.attrib['height'])))
                            bbox_list.append(bbox) # box = [xywh_c]
                            # print('4', four_child.tag, bbox)



            outimgnum += 1
            if bbox_list and outimgnum % 10 == 0:
                imgpath = imgdir + '/' + imgdirname + '/' + 'img' + str((child.attrib.get('num'))).zfill(5) + '.jpg'
                save_path = out_img_path + imgdirname + '_' + 'img' + str((child.attrib.get('num'))).zfill(5) + '.jpg'
                h,w,_ = cv2.imread(imgpath).shape
                newlabel = outlabel + imgdirname + '_' + 'img' + str((child.attrib.get('num'))).zfill(5) +'.txt'
                newf = open(newlabel,'w')
                shutil.copyfile(imgpath, save_path)
                for newbox in bbox_list:
                    newbox[0] = newbox[0] / w
                    newbox[1] = newbox[1] / h
                    newbox[2] = newbox[2] / w
                    newbox[3] = newbox[3] / h
                    newf.write('1' + " " + " ".join([str(a) for a in newbox]) + '\n')
                newf.close()
def uadet_mat2yolo5(input,inpurimgs,output):
    outimgpath = output + 'images/'
    labelpath = output + 'labels/'
    os.makedirs(outimgpath, exist_ok=True)
    os.makedirs(labelpath, exist_ok=True)
    for matfile in os.listdir(input):
            matfilepath = input + matfile
            mat = scio.loadmat(matfilepath)
            import pdb
            pdb.set_trace()
def bit2yolo(input,output):
    outimgpath = output + 'images/'
    labelpath = output + 'labels/'
    os.makedirs(outimgpath, exist_ok=True)
    os.makedirs(labelpath, exist_ok=True)
    # import pdb
    # pdb.set_trace()
    for matfile in [x for x in walkfile(input) if x.endswith('mat')]:
        mat = scio.loadmat(matfile)['VehicleInfo']
        for i , sample in tqdm.tqdm(enumerate(mat)):
            imagename = sample[0][0][0]
            h,w = sample[0][1][0][0],sample[0][2][0][0]
            boxes = sample[0][3][0]
            inimg = input + 'BITVehicle_Dataset/' + imagename
            outimg = outimgpath + imagename
            outlabel = labelpath + imagename.replace('jpg','txt')
            newf = open(outlabel,'w')
            for box in boxes:
                nbox = [box[0][0][0],box[1][0][0],\
                      box[2][0][0],box[3][0][0]]
                newbox = xyxy2xywh_c(nbox)
                newbox[0] = newbox[0] / w
                newbox[1] = newbox[1] / h
                newbox[2] = newbox[2] / w
                newbox[3] = newbox[3] / h
                newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
            newf.close()
            shutil.copyfile(inimg,outimg)
def nui2yolo5(input,output):
    thelist = ["/data/linwang/nuimg/v1.0-train/","/data/linwang/nuimg/v1.0-val/"]
    for labelsdir in thelist:
        # inputimgs = input +  filename
        category = open(labelsdir + 'category.json')
        cates = [json.loads(line) for line in category.readlines()]
        sample_data = open(labelsdir +  'sample_data.json')
        annos = open(labelsdir + 'object_ann.json')
def visdrone2yolo(input,output):
    categories=['pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle', 'awningtricycle','tricycle']
    imgs = [x for x in walkfile(input) if x.endswith('jpg')]
    outped = output + 'vdperson/'
    outveh = output + 'vdveh/'
    os.makedirs(outped + 'images/trains/',exist_ok=True)
    os.makedirs(outped + 'images/valids/',exist_ok=True)
    os.makedirs(outped + 'labels/trains/',exist_ok=True)
    os.makedirs(outped + 'labels/valids/',exist_ok=True)
    os.makedirs(outveh + 'images/trains/',exist_ok=True)
    os.makedirs(outveh + 'images/valids/',exist_ok=True)
    os.makedirs(outveh + 'labels/trains/',exist_ok=True)
    os.makedirs(outveh + 'labels/valids/',exist_ok=True)
    pedsave,vehsave = 0,0
    for img in tqdm.tqdm(imgs):
        pedboxes = []
        vehboxes = []
        label = img.replace('images','annotations').replace('jpg','txt')
        name = img.split(os.sep)[-1].replace('.jpg','')
        if os.path.isfile(label):
            image = cv2.imread(img)
            h,w,_ = image.shape

            with open(label, 'r') as f:
                    lb = np.array([x.strip(',').split(',') for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

            for sample in lb:
                if sample[5] in [1,2]:
                    newbox = np.array(xywh2xywh_c(sample[:4])) / [w,h,w,h]
                    pedboxes.append(newbox)
                elif sample[5] in [4,5,6,7,8,9,10]:
                    newbox = np.array(xywh2xywh_c(sample[:4])) / [w,h,w,h]
                    vehboxes.append(newbox)
            if len(pedboxes) != 0:
                if pedsave % 10 == 0:
                    with open(outped + f'labels/valids/{name}.txt', 'w') as newf:
                        for newbox in pedboxes:
                            newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
                    shutil.copy(img,outped + 'images/valids/')
                else:
                    with open(outped + f'labels/trains/{name}.txt', 'w') as newf:
                        for newbox in pedboxes:
                            newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
                    shutil.copy(img, outped + 'images/trains/')
                pedsave += 1
            if len(vehboxes) != 0:
                if vehsave % 10 == 0:
                    with open(outveh + f'labels/valids/{name}.txt', 'w') as newf:
                        for newbox in vehboxes:
                            newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
                    shutil.copy(img,outveh + 'images/valids/')
                else:
                    with open(outveh + f'labels/trains/{name}.txt', 'w') as newf:
                        for newbox in vehboxes:
                            newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
                    shutil.copy(img, outveh + 'images/trains/')
                vehsave += 1
def visdroneemot2yolo(input,output):
    labels = [x for x in walkfile(input) if x.endswith('txt')]
    outped = output + 'vdperson_test/'
    outveh = output + 'vdveh_test/'
    os.makedirs(outped + 'images/', exist_ok=True)
    os.makedirs(outped + 'labels/', exist_ok=True)
    os.makedirs(outveh + 'images/', exist_ok=True)
    os.makedirs(outveh + 'labels/', exist_ok=True)

    for label in labels:
        with open(label, 'r') as f:
            lb = np.array([x.strip(',').split(',') for x in f.read().strip().splitlines()],
                          dtype=np.float32)  # labels

        dirname = label.split(os.sep)[-1].split('.')[0]
        imgdir = label.replace('annotations','sequences').replace('.txt','/')
        imgs = [x for x in walkfile(imgdir) if x.endswith('jpg')]
        for img in tqdm.tqdm(imgs):
            pedboxes = []
            vehboxes = []
            image = cv2.imread(img)
            h,w,_ = image.shape
            imgname = img.split(os.sep)[-1].replace('.jpg','')
            frameid = int(imgname)
            target = lb[lb[:,0]==frameid]
            # pdb.set_trace()
            for sample in target:
                if sample[7] in [1,2]:
                    b = np.array(xywh2xywh_c(sample[2:6])) / [w,h,w,h]
                    pedboxes.append(b)
            with open(outped  + f'labels/{dirname}_{imgname}.txt', 'w') as newf:
                for newbox in pedboxes:
                    newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
            shutil.copyfile(img,outped  + f'images/{dirname}_{imgname}.jpg')
def debug(inputvideo):
    output_path = './output.avi'
    vc = cv2.VideoCapture(0)
    ret, frame = vc.read()
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # fourcc = cv2.VideoWriter_fourcc('H', 'E', 'V', 'C')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
    while ret:
        vw.write(frame)
        ret, frame = vc.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return
def drawstatistics():

    data_dir ="/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/metadata.xlsx"   # excel dir

    video = ['a001','a002','b001','b002']
    task = ['ped','veh']
    diff = ['hard','normal']
    # end = np.datetime64('2021-02-10')
    # files = glob.glob(data_dir + '/*.xlsx')
    filename = os.path.basename(data_dir).split('.')[0]
    data = pd.read_excel(data_dir, sheet_name=['Sheet2'])
    # import pdb
    # pdb.set_trace()
    header = ['video', 'name', 'label', 'position', 'difficulty', 'alltime',
              'hit(test1)', 'maxcontinue(test1)', 'conf(test1)', 'stability(test1)',
              'hit(test1).1', 'maxcontinue(test2)', 'conf(test2)', 'stability(test2)',
              'hit(test3)', 'maxcontinue(test3)', 'conf(test3)', 'stability(test3)',
              'ps', 'pps']
    data1 = data['Sheet2']
    lens = len(data['Sheet2'][header[1]])
    pan,pah,pbn,pbh,van,vah,vbn,vbh = [],[],[],[],[],[],[],[]      # p,v = ped,veh  a,b videoname h,n difficulty
    for i in range(lens):
        assert data1['video'][i] in video , 'data err'
        assert data1['label'][i] in task , 'data err'
        assert data1['difficulty'][i] in diff , 'data err'
        if data1['label'][i] == 'ped':  #  Divided into 8 classes
            if data1['video'][i]  == 'a001' or data1['video'][i]  == 'a002':
                if data1['difficulty'][i] == 'normal':
                    pan.append(i)
                elif data1['difficulty'][i] == 'hard':
                    pah.append(i)
            elif data1['video'][i]  == 'b001' or data1['video'][i]  == 'b002':
                if data1['difficulty'][i] == 'normal':
                    pbn.append(i)
                elif data1['difficulty'][i] == 'hard':
                    pbh.append(i)
        elif data1['label'][i] == 'veh':
            if data1['video'][i]  == 'a001' or data1['video'][i]  == 'a002':
                if data1['difficulty'][i] == 'normal':
                    van.append(i)
                elif data1['difficulty'][i] == 'hard':
                    vah.append(i)
            elif data1['video'][i]  == 'b001' or data1['video'][i]  == 'b002':
                if data1['difficulty'][i] == 'normal':
                    vbn.append(i)
                elif data1['difficulty'][i] == 'hard':
                    vbh.append(i)
    divided = [pan,pah,pbn,pbh,van,vah,vbn,vbh]
    results = []
    for num , part in enumerate(divided):
        totaltime = 0
        totalhit1 = 0
        tp1 = 0
        totalhit2 = 0
        tp2 = 0
        totalhit3 = 0
        tp3 = 0
        count = len(part)
        stability1,stability2,stability3 = 0,0,0
        result =[]
        test1,test2,test3 =[],[],[]
        for x,index in enumerate(part):
            if num <4 :
                conf = 1
            else:
                conf = 20
            totaltime += data1['alltime'][index]
            totalhit1 += data1['hit(test1)'][index]
            stability1 += data1['stability(test1)'][index]
            if data1['maxcontinue(test1)'][index] > conf :
                    tp1 += 1
            totalhit2 += data1['hit(test2)'][index]
            stability2 += data1['stability(test2)'][index]
            if data1['maxcontinue(test2)'][index] > conf:
                    tp2 += 1
            totalhit3 += data1['hit(test3)'][index]
            stability3 += data1['stability(test3)'][index]
            if data1['maxcontinue(test3)'][index] > conf:
                    tp3 += 1
            result = [totaltime,count,totalhit1 , tp1/count,stability1/count,totalhit2 , tp2/count,stability2/count,totalhit3 , tp3/count,stability3/count]
        results.append(result)
    print('pan,pah,pbn,pbh,van,vah,vbn,vbh:',results)

def multi_videos_out_clean(savepath):
    files = [savepath + file for file in os.listdir(savepath)]
    for dir in files:
        newimagespath = dir + '/newbgimages/'
        os.makedirs(newimagespath,exist_ok=True)
        if 'veh' in dir:
            labelspath = dir + '/labels'
            imgspath = dir + '/bgimages'
            imgs = walkfile(imgspath)
            labels = walkfile(labelspath)
            print(len(imgs),len(labels))
            for sample in tqdm.tqdm(labels):
                # import pdb
                # pdb.set_trace()
                imgpath = sample.replace('labels','images').replace('xml','jpg')
                if os.path.isfile(imgpath):
                    shutil.copy(imgpath,newimagespath)
def xml_labelme2json(input,output):
    '''
    xml:imgme old label tool
    json:labelme new label tool
    json:yolo3 format
    '''
    imgs = [x for x in walkfile(input) if x.endswith('jpg')]
    jsonout_v = output + 'test_v.json'
    jsonout_p = output + 'test_p.json'
    jsonout_a_h_v = output + 'test_a_h_v.json'
    jsonout_a_n_v = output + 'test_a_n_v.json'
    jsonout_a_h_p = output + 'test_a_h_p.json'
    jsonout_a_n_p = output + 'test_a_n_p.json'
    jsonout_b_h_v = output + 'test_b_h_v.json'
    jsonout_b_n_v = output + 'test_b_n_v.json'
    jsonout_b_h_p = output + 'test_b_h_p.json'
    jsonout_b_n_p = output + 'test_b_n_p.json'
    list_v,list_p,list_a_h_v,list_a_n_v,list_a_h_p,list_a_n_p,list_b_h_v,list_b_n_v,list_b_h_p,list_b_n_p = [],[],[],[],[],[],[],[],[],[]
    countjson,countxml = 0,0
    os.makedirs(output + 'sample/',exist_ok=True)
    for imgpath in tqdm.tqdm(imgs):
        if 'veh' in imgpath:
            jsonpath = imgpath.replace('jpg','json').replace('bgimages','jsonlabels')
            if os.path.isfile(jsonpath):
                f = open(jsonpath, 'r', encoding='UTF-8')
                lines = f.read()
                info = json.loads(lines)
                target = []
                for sample in info.get('shapes',[]):
                        x1,y1,x2,y2 = min(sample['points'][0][0],sample['points'][1][0]),min(sample['points'][0][1],sample['points'][1][1]),max(sample['points'][0][0],sample['points'][1][0]),max(sample['points'][0][1],sample['points'][1][1])
                        target.append([int(x1),int(y1),int(x2),int(y2),0])
                if target != []:
                        if countjson < 50:
                            img = cv2.imread(imgpath)
                            for box in target:
                                cv2.rectangle(img,(box[1],box[2]),(box[3],box[4]),(255,0,0),2)
                            cv2.imwrite(output + 'sample/' + str(countjson) + 'vehjson.jpg',img)
                            countjson += 1
                        annot_dict = [('input', imgpath), ('target', target)]
                        dict1 = dict(annot_dict)
                        list_v.append(dict1)
                        if 'hard' in imgpath and 'A' in imgpath:
                            list_a_h_v.append(dict1)
                        elif 'normal' in imgpath and 'A' in imgpath:
                            list_a_n_v.append(dict1)
                        elif 'hard' in imgpath and 'B' in imgpath:
                            list_b_h_v.append(dict1)
                        elif 'normal' in imgpath and 'B' in imgpath:
                            list_b_n_v.append(dict1)
        elif 'ped' in imgpath:  # only ped have xmlfiles
            jsonpath = imgpath.replace('jpg', 'json').replace('images', 'jsonlabels')
            xmlpath = imgpath.replace('jpg', 'xml').replace('images', 'labels')
            if os.path.isfile(jsonpath):
                f = open(jsonpath, 'r', encoding='UTF-8')
                lines = f.read()
                info = json.loads(lines)
                target = []
                for sample in info.get('shapes',[]):
                        x1, y1, x2, y2 = min(sample['points'][0][0], sample['points'][1][0]), min(
                            sample['points'][0][1], sample['points'][1][1]), max(sample['points'][0][0],
                                                                                 sample['points'][1][0]), max(
                            sample['points'][0][1], sample['points'][1][1])
                        target.append([int(x1), int(y1), int(x2), int(y2), 0])
                if target != []:
                        annot_dict = [('input', imgpath), ('target', target)]
                        dict1 = dict(annot_dict)
                        list_p.append(dict1)
                        if 'hard' in imgpath and 'A' in imgpath:
                            list_a_h_p.append(dict1)
                        elif 'normal' in imgpath and 'A' in imgpath:
                            list_a_n_p.append(dict1)
                        elif 'hard' in imgpath and 'B' in imgpath:
                            list_b_h_p.append(dict1)
                        elif 'normal' in imgpath and 'B' in imgpath:
                            list_b_n_p.append(dict1)
            elif os.path.isfile(xmlpath):
                # import pdb
                # pdb.set_trace()
                et = ET.parse(xmlpath)
                element = et.getroot()
                element_objs = element.findall('object')
                target = []
                for element_obj in element_objs:
                    box = []
                    box.append(int(element_obj.find('bndbox').find('xmin').text))
                    box.append(int(element_obj.find('bndbox').find('ymin').text))
                    box.append(int(element_obj.find('bndbox').find('xmax').text))
                    box.append(int(element_obj.find('bndbox').find('ymax').text))
                    box.append(0)
                    target.append(box)
                if target != []:
                    if countxml < 50:
                        img = cv2.imread(imgpath)
                        for box in target:
                            cv2.rectangle(img,(box[1],box[2]),(box[3],box[4]),(255,0,0),2)
                        cv2.imwrite(output + 'sample/' + str(countxml) + 'xml.jpg', img)
                        countxml += 1
                    annot_dict = [('input', imgpath), ('target', target)]
                    dict1 = dict(annot_dict)
                    list_p.append(dict1)
                    if 'hard' in imgpath and 'A' in imgpath:
                        list_a_h_p.append(dict1)
                    elif 'normal' in imgpath and 'A' in imgpath:
                        list_a_n_p.append(dict1)
                    elif 'hard' in imgpath and 'B' in imgpath:
                        list_b_h_p.append(dict1)
                    elif 'normal' in imgpath and 'B' in imgpath:
                        list_b_n_p.append(dict1)

    with open(jsonout_v, 'w', encoding='utf-8') as f:
        for sample in list_v:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_p, 'w', encoding='utf-8') as f:
        for sample in list_p:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_a_h_v, 'w', encoding='utf-8') as f:
        for sample in list_a_h_v:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_a_n_v, 'w', encoding='utf-8') as f:
        for sample in list_a_n_v:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_a_h_p, 'w', encoding='utf-8') as f:
        for sample in list_a_h_p:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_a_n_p, 'w', encoding='utf-8') as f:
        for sample in list_a_n_p:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_b_h_v, 'w', encoding='utf-8') as f:
        for sample in list_b_h_v:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_b_n_v, 'w', encoding='utf-8') as f:
        for sample in list_b_n_v:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_b_h_p, 'w', encoding='utf-8') as f:
        for sample in list_b_h_p:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonout_b_n_p, 'w', encoding='utf-8') as f:
        for sample in list_b_n_p:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return
def labelme2json(input,output):
    conepath = "/data/linwang/ITS/task4_ori/task3/labels/"
    pedpath = "/data/linwang/ITS/task4_ori/task3/ignores/"
    imgspath = "/data/linwang/ITS/task4_ori/task3/frames/"
    output = "/data/linwang/ITS/task4_ori/task3/outjsons/"
    videos =os.listdir(imgspath)
    count = [0,0,0,0,0,0] # cone num ,ped num,biker,ref num,ignores,imgsnum
    for videoname in tqdm.tqdm(videos):
        # pdb.set_trace()
        list_out = []
        outjson = output + videoname + '.json'
        imgs = [x for x in walkfile(imgspath + videoname) if x.endswith('jpg')]
        for imgpath in imgs:
            videoname = imgpath.split(os.sep)[-2]
            imgname = imgpath.split(os.sep)[-1].replace('.jpg','')
            framenum = int(imgpath.split(os.sep)[-1].replace('.jpg','').split('_')[-1])
            conejsonpath = os.path.join(conepath ,videoname , imgname +'.json')
            pedjsonpath = os.path.join(pedpath ,videoname , imgname + '.json')
            if os.path.isfile(conejsonpath) or os.path.isfile(pedjsonpath):
                target = []
                count[5] += 1
                if os.path.isfile(conejsonpath):
                    f = open(conejsonpath, 'r', encoding='UTF-8')
                    lines = f.read()
                    info = json.loads(lines)

                    for sample in info.get('shapes', []):
                        x1, y1, x2, y2 = min(sample['points'][0][0], sample['points'][1][0]), min(sample['points'][0][1],
                                                                                                  sample['points'][1][
                                                                                                      1]), max(
                            sample['points'][0][0], sample['points'][1][0]), max(sample['points'][0][1],
                                                                                 sample['points'][1][1])
                        count[int(sample['label'])] += 1
                        target.append([(x1), (y1), (x2), (y2), int(sample['label'])])
                if os.path.isfile(pedjsonpath):
                        f = open(pedjsonpath, 'r', encoding='UTF-8')
                        lines = f.read()
                        info = json.loads(lines)

                        for sample in info.get('shapes', []):
                            x1, y1, x2, y2 = min(sample['points'][0][0], sample['points'][1][0]), min(
                                sample['points'][0][1],
                                sample['points'][1][
                                    1]), max(
                                sample['points'][0][0], sample['points'][1][0]), max(sample['points'][0][1],
                                                                                     sample['points'][1][1])
                            count[int(sample['label'])] += 1
                            target.append([(x1), (y1), (x2), (y2), int(sample['label'])])
                ignores = np.array(target)
                annot_dict = [('input', framenum), ('target', target)]
                dict1 = dict(annot_dict)
                list_out.append(dict1)
        with open(outjson, 'w', encoding='utf-8') as f:
            for sample in list_out:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(count)
def labelme2txt(input,output):
    '''
    labelme:labelme format json
    txt:yolo5 format
    '''
    outlabels = output + 'labels/'
    outimgs = output + 'images/'
    samplesout = output + 'sample/'
    os.makedirs(outimgs,exist_ok=True)
    os.makedirs(outlabels, exist_ok=True)
    os.makedirs(samplesout, exist_ok=True)
    imgs = [x for x in walkfile(input) if x.endswith('jpg')]
    for imgpath in tqdm.tqdm(imgs):
        # import pdb
        # pdb.set_trace()
        videoname = imgpath.split(os.sep)[-2]
        jsonpath = imgpath.replace('jpg', 'json').replace('images', 'label_json')
        if os.path.isfile(jsonpath):
            f = open(jsonpath, 'r', encoding='UTF-8')
            lines = f.read()
            info = json.loads(lines)
            target = []
            for sample in info.get('shapes', []):
                x1, y1, x2, y2 = min(sample['points'][0][0], sample['points'][1][0]), min(
                    sample['points'][0][1], sample['points'][1][1]), max(sample['points'][0][0],
                                                                         sample['points'][1][0]), max(
                    sample['points'][0][1], sample['points'][1][1])
                target.append([int(x1), int(y1), int(x2), int(y2), 0])
            if target != []:
                newf = open(outlabels+os.path.basename(jsonpath).replace('json','txt'),'w')
                img = cv2.imread(imgpath)
                h,w,_ = img.shape
                for box in target:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    nbox = box[:4]
                    newbox = xyxy2xywh_c(nbox)
                    newbox[0] = newbox[0] / w
                    newbox[1] = newbox[1] / h
                    newbox[2] = newbox[2] / w
                    newbox[3] = newbox[3] / h
                    newf.write('0' + " " + " ".join([str(a) for a in newbox]) + '\n')
                # cv2.imwrite(output + 'sample/' + os.path.basename(imgpath), img)
                # shutil.copy(imgpath,outimgs)
def labelsize(input):
    for key,jsonpath in input.items():
        total_h = 0
        total_w = 0
        total_num = 0
        fid =  open(jsonpath, "r")
        samples = [json.loads(line) for line in fid]
        for sample in tqdm.tqdm(samples):
            label_h = 0
            label_w = 0
            label_num = 0
            if 'A001' in sample['input'] or 'A002' in sample['input']:
                h,w = 1080,1920
            elif 'B001ped' in sample['input'] or 'B002ped' in sample['input']:
                h,w = 3000,4000
            elif 'B001veh' in sample['input'] or 'B002veh' in sample['input']:
                h,w = 1500,2000
            else:
                h,w,_ = cv2.imread(sample['input']).shape

            for box in sample['target']:
                label_h += (box[3] - box[1])/h
                label_w += (box[2] - box[0])/w
                label_num += 1
            total_h += label_h/label_num
            total_w += label_w/label_num
            total_num += label_num
        print(key,total_h/len(samples) ,total_w/len(samples) ,total_num)

def rmpedlabels(input):
    errlist = ['set','Fudan','Penn']
    files = [x for x in walkfile(input) if x.endswith('jpg')]
    for file in tqdm.tqdm(files):
        label = file.replace('images','labels').replace('jpg','txt')
        for err in errlist:
            if err in file:
                os.remove(file)
                print(file)


def preprocess(image, label,size=640):
        h,w,_ = image.shape
        if image.shape[0] > image.shape[1]:
            width, height = int(image.shape[1] * size / image.shape[0]), size
            pad = ((0, 0), (0, size - width), (0, 0))
        else:
            width, height = size, int(image.shape[0] * size / image.shape[1])
            pad = ((0, size - height), (0, 0), (0, 0))
        label[:, 1:] = label[:, 1:] * [width / size, height / size, width / size, height / size]
        image = cv2.resize(image, (width, height))
        image = np.pad(image, pad)
        return image,label
def zoomout(input,output,thres = 0.5):
    imglist = [file for file in walkfile(input + 'images/') if file.endswith('jpg')]
    labellist = [file for file in walkfile(input + 'labels/') if file.endswith('txt')]

    resizelist = []
    orilist = []
    out_rmlabel = output + 'rm_only/'
    out_resizelabel = output + 'rs_withrm/'
    out_rs_without_rm = output + 'rs_withoutrm/'
    os.makedirs(out_rmlabel,exist_ok=True)
    os.makedirs(out_resizelabel, exist_ok=True)
    os.makedirs(out_rs_without_rm, exist_ok=True)
    makeset(out_rmlabel)
    makeset(out_resizelabel)
    makeset(out_rs_without_rm)
    for label in tqdm.tqdm(labellist):
        target_list = []
        tem = label.replace('labels','images')
        imgpath = tem.replace('txt','jpg')
        if os.path.isfile(imgpath):
            with open(label, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            _resize = False
            # for j, x in enumerate(lb):
            #     c = int(x[0])  # class
            #     if x[3] > thres or x[4] > thres:
            #         _resize = True
            try:
                if sum(lb[:,3]>thres) > 0 or sum(lb[:,4]>thres) > 0:
                    _resize = True
            except:
                continue
            resizelist.append(label) if _resize else orilist.append(label)
    # pdb.set_trace()
    print(len(orilist),len(resizelist))
    for i,orilabel in tqdm.tqdm(enumerate(orilist)):
        imgpath = orilabel.replace('labels', 'images').replace('txt', 'jpg')
        if i % 9 == 0:
            shutil.copy(orilabel,out_rmlabel + 'labels/valids/')
            shutil.copy(orilabel, out_resizelabel + 'labels/valids/')
            shutil.copy(orilabel, out_rs_without_rm + 'labels/valids/')
            shutil.copy(imgpath, out_rmlabel + 'images/valids/')
            shutil.copy(imgpath, out_resizelabel + 'images/valids/')
            shutil.copy(imgpath, out_rs_without_rm + 'images/valids/')
        else:
            shutil.copy(orilabel, out_rmlabel + 'labels/trains/')
            shutil.copy(orilabel, out_resizelabel + 'labels/trains/')
            shutil.copy(orilabel, out_rs_without_rm + 'labels/trains/')
            shutil.copy(imgpath, out_rmlabel + 'images/trains/')
            shutil.copy(imgpath, out_resizelabel + 'images/trains/')
            shutil.copy(imgpath, out_rs_without_rm + 'images/trains/')
    _oneimg = []
    savenum = 0
    random.shuffle(resizelist)
    for i,rslabel in tqdm.tqdm(enumerate(resizelist)):
        imgpath = rslabel.replace('labels', 'images').replace('txt', 'jpg')
        _oneimg.append(rslabel)
        if i%9 ==0 :
            shutil.copy(imgpath, out_rs_without_rm + 'images/valids/')
            shutil.copy(rslabel, out_rs_without_rm + 'labels/valids/')
        else:
            shutil.copy(imgpath, out_rs_without_rm + 'images/trains/')
            shutil.copy(rslabel, out_rs_without_rm + 'labels/trains/')
        if len(_oneimg) == 4:
            newimg,newlabel = _zoomout(_oneimg,size = 640)

            # pdb.set_trace()
            # img = checktxt(newimg,newlabel)
            _oneimg = []
            saveimg = 'zoomout'+str(savenum).zfill(8) + '.jpg'
            if savenum%9 == 0:
                # cv2.imwrite(out_resizelabel + 'images/valids/' +saveimg,newimg)
                cv2.imwrite(out_rs_without_rm + 'images/valids/' + saveimg, newimg)
                # shutil.copy(out_resizelabel + 'images/valids/' +saveimg,out_rs_without_rm + 'images/valids/' + saveimg)
                with open(out_rs_without_rm + 'labels/valids/'+saveimg.replace('jpg','txt'),'w') as newf:
                    for sam in newlabel:
                        newf.write('0' + " " + " ".join([str(a) for a in sam[1:]]) + '\n')
                # shutil.copy(out_rs_without_rm + 'labels/valids/'+saveimg.replace('jpg','txt'),out_resizelabel + 'labels/valids/'+saveimg.replace('jpg','txt'))
            else:
                # cv2.imwrite(out_resizelabel + 'images/trains/' + saveimg, newimg)
                cv2.imwrite(out_rs_without_rm + 'images/trains/' + saveimg, newimg)
                # shutil.copy(out_resizelabel + 'images/trains/' + saveimg,out_rs_without_rm + 'images/trains/' + saveimg)
                with open(out_rs_without_rm + 'labels/trains/'+saveimg.replace('jpg','txt'),'w') as newf:
                    for sam in newlabel:
                        newf.write('0' + " " + " ".join([str(a) for a in sam[1:]]) + '\n')
                # shutil.copy(out_rs_without_rm + 'labels/trains/'+saveimg.replace('jpg','txt'),out_resizelabel + 'labels/trains/'+saveimg.replace('jpg','txt'))
            savenum += 1
def _zoomout(oneimg,size = 640):

        newimg = np.zeros((int(size*np.sqrt(len(oneimg))), int(size*np.sqrt(len(oneimg))), 3), np.uint8)
        newlabel = []
        for i,sample in enumerate(oneimg):
            # pdb.set_trace()
            imgpath = sample.replace('labels', 'images').replace('txt', 'jpg')
            img = cv2.imread(imgpath)
            h,w,_ = img.shape
            with open(sample,'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            img,lb = preprocess(img,lb,size)
            y_num,x_num = int(i/np.sqrt(len(oneimg))),int(i%np.sqrt(len(oneimg)))
            newimg[y_num*size:(y_num+1)*size,x_num*size:(x_num+1)*size] = img

            lb[:,1] = lb[:,1]/np.sqrt(len(oneimg)) + x_num*(1./np.sqrt(len(oneimg)))
            lb[:, 2] = lb[:, 2]/np.sqrt(len(oneimg)) + y_num * (1. / np.sqrt(len(oneimg)))
            lb[:, 3] = lb[:, 3] * (1. / np.sqrt(len(oneimg)))
            lb[:, 4] = lb[:, 4] * (1. / np.sqrt(len(oneimg)))
            lb = np.unique(lb,axis=0)
            for _x in lb:
                newlabel.append(_x)
        return newimg,newlabel
def checktxt(img,label):
    if isinstance(img,str) and os.path.isfile(img):
        img = cv2.imread(img)
    if isinstance(img,str) and os.path.isfile(label):
        with open(label,'r') as f:
            label = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    h,w,_ = img.shape

    label = np.unique(label, axis=0)
    try:
        # label = label[label[:,0] == 0]
        label = label[label[:,1] > 0.]
        label = label[label[:, 1] < 1.]
        label = label[label[:, 3] > 0.]
        label = label[label[:, 3] < 1.]
        label = label[label[:, 4] > 0.]
        label = label[label[:, 4] < 1.]
        label = label[label[:, 2] > 0.]
        newlabel = label[label[:, 2] < 1.]
    except:
        return img,label
    # for sample in label:
    #     if sample[0] == 0 and sample[1] > 0. and sample[1] < 1.  and sample[2] > 0. and sample[2] < 1.:
    #         newlabel.append(sample)
    newlabel = np.maximum(newlabel, -newlabel)
    # pdb.set_trace()
    for sample in label:
        newbox = xywh_c2xyxy(sample[1:])
        cv2.rectangle(img,(int(newbox[0]*w),int(newbox[1]*h)),(int(newbox[2]*w),int(newbox[3]*h)),(255,0,0),2)
    return img,newlabel
def makeset(input):
    os.makedirs(input + 'images/trains/',exist_ok=True)
    os.makedirs(input + 'images/valids/', exist_ok=True)
    os.makedirs(input + 'labels/trains/',exist_ok=True)
    os.makedirs(input + 'labels/valids/', exist_ok=True)
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (max(int(x[0]),0), max(int(x[1]),0)), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 2)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def plot_txt(root,out):

    images = [file for file in walkfile(root + 'images/') if file.endswith('jpg')]
    savedir = out + "show/"
    os.makedirs(savedir, exist_ok=True)
    for filename in tqdm.tqdm(images):
        name = filename.split(os.sep)[-1]
        annotation = filename.replace('images','labels').replace('jpg','txt')
        save = False
        if os.path.isfile(annotation):
            image = cv2.imread(filename)
            h, w, _ = image.shape

            with open(annotation, "r") as f:
                label = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                for sample in label:
                    if True:
                        save = True
                        # pdb.set_trace()
                        box = sample[1:] * [w,h,w,h]
                        newbox = xywh_c2xyxy(box)
                        plot_one_box(newbox,image,color = (0,0,255),label=str(int(sample[0])),line_thickness=1)
        if save:
            cv2.imwrite(savedir + name , image)
def drawmevavideo(videospath,labelspath,outpath):
    labels = [file for file in walkfile(labelspath) if file.endswith('txt')]
    videos = [file for file in walkfile(videospath) if file.endswith('avi')]
    for videopath in tqdm.tqdm(videos):
        videoname = videopath.split(os.sep)[-1].replace('.r13.avi','')
        labelpath = videopath.replace(videospath,labelspath).replace('.r13.avi','')
        _labels = [file for file in walkfile(labelpath) if file.endswith('txt')]
        if len(_labels) > 1000:
            # pdb.set_trace()
            cap = cv2.VideoCapture(videopath)
            ret, frame = cap.read()
            framenum = 0
            h,w,_ = frame.shape
            savepath = outpath + videoname + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
            outvideo = cv2.VideoWriter(savepath, fourcc, 30, (w, h))
            while ret:
                _labelpath = labelpath +os.sep+ videoname + f'_{framenum}.txt'
                if os.path.isfile(_labelpath) :
                    with open(_labelpath, "r") as f:
                        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                    for sample in lb:
                        plot_one_box(sample[1:5],frame,label=str(sample[0]),line_thickness=2)
                outvideo.write(frame)
                ret,frame = cap.read()
                framenum += 1
            cap.release()
            outvideo.release()
def json2labelme(input,output):
    '''
    json:yolo3 format
    labelme:new label tool json format
    '''
    jsons = [x for x in walkfile(input) if x.endswith('json')]
    print(len(jsons))
    # pdb.set_trace()
    count = 0
    for jsonpath in tqdm.tqdm(jsons):
        name = jsonpath.split(os.sep)[-1].replace('.json','')
        outpath = output + name + os.sep
        imgspath = '/data/linwang/ITS/outimages_linwang/'
        if os.path.isdir(outpath):
            f = open(jsonpath, 'r', encoding='UTF-8')
            lines = f.readlines()
            for i, line in enumerate(lines):
                info = json.loads(line)
                outdict = {"version":"4.5.9",
                           "flags":{},
                           "shapes":[],
                           "imagePath":"..\\..\\" + name +'\\' + name + '_{}'.format(str(info['input']).zfill(8)) + '.jpg',
                           "imageData":None,
                           "imageHight":1080,
                           "imageWidth":1920
                           }
                for sample in info['target']:
                    labeldict = {"label":str(int(sample[4])),
                                 "points":[[sample[0],sample[1]],[sample[2],sample[3]]],
                                 "group_id":None,
                                 "shape_type":"rectangle",
                                 "flags":{}}
                    outdict["shapes"].append(labeldict)
                outjsonname = outpath + name + '_{}'.format(str(info['input']).zfill(8)) + '.json'
                imgname = imgspath + name + os.sep + name + '_{}'.format(str(info['input']).zfill(8)) + '.jpg'
                if os.path.isfile(imgname):
                    count += 1
                    with open(outjsonname,'w') as f:
                        f.write(json.dumps(outdict, ensure_ascii=False))
    print(count)
def yuli2labelme(input,output):
    for file in [x for x in walkfile(input) if x.endswith('json')]:
        imgpath = file.replace('annos','frames').replace('json','jpg')
        f = open(file, 'r', encoding='UTF-8')
        lines = f.readlines()
        imgname = imgpath.split(os.sep)[-1].replace('.jpg','')
        videoname = imgpath.split(os.sep)[-2]
        outdir = output + videoname + '/'
        os.makedirs(outdir,exist_ok=True)
        for i, line in enumerate(lines):

            info = json.loads(line)
            outdict = {"version": "4.5.9",
                           "flags": {},
                           "shapes": [],
                           "imagePath": "..\\..\\frames\\" + videoname + '\\' + imgname + '.jpg',
                           "imageData": None,
                           "imageHight": 1080,
                           "imageWidth": 1920
                           }
            for sample in info['targets']:
                    labeldict = {"label": str(int(sample[0])),
                                 "points": [[sample[1], sample[2]], [sample[3], sample[4]]],
                                 "group_id": None,
                                 "shape_type": "rectangle",
                                 "flags": {}}
                    outdict["shapes"].append(labeldict)
            outjsonname = outdir + imgname +  '.json'
            if os.path.isfile(imgpath):

                    with open(outjsonname, 'w') as f:
                        f.write(json.dumps(outdict, ensure_ascii=False))
def drawrec_resize(input):
        dic = {
            'bouncingballs':{'005':[172,103],'019':[170,131]},
            'hellwarrior':{'000':[224,330],'003':[98,295]},
            'hook':{'001':[228,122],'007':[223,164]},
            'jumpingjacks':{'007':[240,211],'017':[273,253]},
            'lego':{'001':[160,186],'003':[163,173]},
            'mutant':{'006':[60,184],'009':[180,152]},
            'standup':{'001':[174,188],'002':[206,258]},
            'trex':{'006':[265,60],'007':[140,212]},
        }
        for  _file in  os.listdir(input):
            if os.path.isdir(os.path.join(input,_file)):
                filedir = _file
                for i,filename in enumerate(os.listdir(os.path.join(input,_file))):
                    fileindex = filename[:3]
                    if filename[0] == '.':continue
                    filepath = os.path.join(input,_file,filename)
                    img = cv2.imread(filepath)
                    # pdb.set_trace()
                    # cutlist = dic[filedir][fileindex]
                    cutlist = [int(filename.split(',')[0]),int(filename.split('.')[0].split(',')[1])]
                    # pdb.set_trace()
                    cutimg = copy.deepcopy(img)[cutlist[1]:cutlist[1]+50,cutlist[0]:cutlist[0]+50]
                    cv2.rectangle(img,tuple(cutlist),(cutlist[0]+50,cutlist[1]+50),(0,0,255),1)
                    resized = cv2.resize(cutimg,(150,150),interpolation = cv2.INTER_AREA)
                    cv2.rectangle(resized,(0,0),(149,149),(0,0,0),1)
                    h,w,_ = cutimg.shape
                    if filedir == 'trex':
                        img[250:400,250:400] = resized
                    else:
                        img[0:150,250:400] = resized
                    os.makedirs(os.path.join('dyout',_file),exist_ok=True)
                    cv2.imwrite(os.path.join('dyout',_file,filename),img)
def txt2txt(labelspath,imgspath):
    imgs_dir = [x for x in walkfile(imgspath) if x.endswith('jpg')]
    labels_dir = [x for x in walkfile(labelspath) if x.endswith('txt')]
    print(len(imgs_dir),len(labels_dir))
    # import pdb
    # pdb.set_trace()
    for label in tqdm.tqdm(labels_dir):
        tem = label.replace('labels','images')
        imgpath = tem.replace('txt','jpg')
        if os.path.isfile(imgpath):
            # img = cv2.imread(imgpath)
            # h,w,_ = img.shape
            with open(label, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            # lb[:,0] = 0
            newf = open(label.replace('labels','single_labels'),'w') 
            for box in lb:  
                newf.write('0' + " " + " ".join([str(a) for a in box[1:]]) + '\n')
            newf.close()
            
if __name__ == '__main__':


    input  =  {
            # 'allped':"/data/linwang/all_ped/labels/ped_val.json",
            'a_h_p': "/home/linwang/dataset/anomaly_test/test_a_h_p.json",
            # 'a_n_p': "/home/linwang/dataset/anomaly_test/test_a_n_p.json",
            # 'b_h_p':"/home/linwang/dataset/anomaly_test/test_b_h_p.json",
            # 'b_n_p':"/home/linwang/dataset/anomaly_test/test_b_n_p.json",
            # 'test_p':"/home/linwang/dataset/anomaly_test/test_p.json",
        }

    pdb.set_trace()
    # label = open(file,'r')
    # inputpath ="/data/linwang/all_veh/"
    # imagelist = [_ for _ in walkfile(inputpath) if _.endswith('jpg')]
    # print(len(imagelist))
    # mul = multiprocess()
    # configs = mul.getconfig(videospath='/home/linwang/dataset/patchcore-data/',saveimgs='/home/linwang/dataset/patchcore-data/patcgcore-data-images/',multinum=4)
    # mul.forward()
    from PIL import Image
    path1 = '/home/linwang/dataset/patchcore-data/Composite/dataset-bg/test/'
    images1 = [i for i in walkfile(path1) if i.endswith('jpg') and 'good' not in i ]
    path2 = "/home/linwang/dataset/patchcore-data/Composite/dataset-bg/train/"
    images2 = [i for i in walkfile(path2) if i.endswith('jpg') ]
    print(len(images1),len(images2))
    for i in images1:
        outpath = i.replace('test','ground_truth')
        img = Image.new('RGB', (4000,3000), color='white')
        os.makedirs(os.path.dirname(outpath),exist_ok=True)
        img.save(outpath)


