import os
import sys
import cv2
from itertools import islice
from xml.dom.minidom import Document
from collections import defaultdict

src_img_dir = 'D:/Thunder/downloads/datasets_fusai/datasets/train/'
src_txt_dir = 'D:/Thunder/downloads/datasets_fusai/datasets/train.txt'
src_xml_dir = 'D:/Thunder/downloads/datasets_fusai/datasets/xml/'

img_labels = defaultdict(list)
f = open(src_txt_dir, 'r')
for i in open(src_txt_dir):
    line = f.readline().strip('\n')
    x = line.split('.jpg ')
    key = x[0]
    value = x[1]
    img_labels[key].append(value)

for key in img_labels:
    if(len(img_labels[key])>1):
        print key
        print img_labels[key]

    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode('Image'))
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(key))
    annotation.appendChild(filename)

    source = doc.createElement('source')
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('Baidu'))
    source.appendChild(database)
    annotation.appendChild(source)

    # obtain the size of the image
    imagefile = src_img_dir + key + '.jpg'
    img = cv2.imread(imagefile)
    imgSize = img.shape

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(imgSize[1])))
    size.appendChild(width)
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(imgSize[0])))
    size.appendChild(height)
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(imgSize[2])))
    size.appendChild(depth)
    annotation.appendChild(size)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode(str(0)))
    annotation.appendChild(segmented)

    # write the coordinates of the b-box
    for b_box in list(img_labels[key]):
        #print b_box
        x = b_box.split(' ')
        if(int(x[1])<0):
            x[1] = '0'
        if (int(x[2]) < 0):
            x[2] = '0'

        object = doc.createElement('object')
        name = doc.createElement('name')
        name.appendChild(doc.createTextNode('n' + x[0]))
        object.appendChild(name)

        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(x[1]))
        bndbox.appendChild(xmin)
        object.appendChild(bndbox)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(x[2]))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(x[3]))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(x[4]))
        bndbox.appendChild(ymax)
        annotation.appendChild(object)

    with open(src_xml_dir + key + '.xml', 'w') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))





