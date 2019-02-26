#!/usr/bin/python3

#导入xml包
import os
import xml.etree.ElementTree as ET

files = "xml/"

for file in os.listdir(files):
    tree = ET.parse("xml/"+file)
    root = tree.getroot()  # 获取根节点
    # root[0].text = 'VOC2012'
    # filename = root[1].text + '.jpg'
    # root[1].text = filename
    # root[5][0].text = str(root[5][0].text[1:])
    tree.write('VOCdevkit/VOC2012/Annotations/'+file, xml_declaration=False, encoding="utf-8")
    # print(root[5][0].text[1:])
