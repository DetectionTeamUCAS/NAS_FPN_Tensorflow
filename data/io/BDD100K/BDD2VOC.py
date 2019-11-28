import json
import os
import cv2
from xml.dom.minidom import Document
import xml.dom.minidom

label_map = {'bus': 1, 'traffic light': 2, 'traffic sign': 3, 'person': 4, 'bike': 5,
             'truck': 6, 'motor': 7, 'car': 8, 'train': 9, 'rider': 10}
FLAG = ['train', 'val']


def write_xml(save_path, name, box_list, label_list, w, h, d):

    # dict_box[filename]=json_dict[filename]
    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    foldername = doc.createElement("folder")
    foldername.appendChild(doc.createTextNode("JPEGImages"))
    root.appendChild(foldername)

    nodeFilename = doc.createElement('filename')
    nodeFilename.appendChild(doc.createTextNode(name))
    root.appendChild(nodeFilename)

    pathname = doc.createElement("path")
    pathname.appendChild(doc.createTextNode("xxxx"))
    root.appendChild(pathname)

    sourcename=doc.createElement("source")

    databasename = doc.createElement("database")
    databasename.appendChild(doc.createTextNode("Unknown"))
    sourcename.appendChild(databasename)

    annotationname = doc.createElement("annotation")
    annotationname.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(annotationname)

    imagename = doc.createElement("image")
    imagename.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(imagename)

    flickridname = doc.createElement("flickrid")
    flickridname.appendChild(doc.createTextNode("0"))
    sourcename.appendChild(flickridname)

    root.appendChild(sourcename)

    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodewidth.appendChild(doc.createTextNode(str(w)))
    nodesize.appendChild(nodewidth)
    nodeheight = doc.createElement('height')
    nodeheight.appendChild(doc.createTextNode(str(h)))
    nodesize.appendChild(nodeheight)
    nodedepth = doc.createElement('depth')
    nodedepth.appendChild(doc.createTextNode(str(d)))
    nodesize.appendChild(nodedepth)
    root.appendChild(nodesize)

    segname = doc.createElement("segmented")
    segname.appendChild(doc.createTextNode("0"))
    root.appendChild(segname)

    for (box, label) in zip(box_list, label_list):

        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(str(label)))
        nodeobject.appendChild(nodename)
        nodebndbox = doc.createElement('bndbox')
        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(box[0])))
        nodebndbox.appendChild(nodex1)
        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(box[1])))
        nodebndbox.appendChild(nodey1)
        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(box[2])))
        nodebndbox.appendChild(nodex2)
        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(box[3])))
        nodebndbox.appendChild(nodey2)

        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(save_path, 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


for flag in FLAG:
    BDD_path = '/BDD100K/bdd100k/'
    BDD_labels_dir = os.path.join(BDD_path, 'labels/bdd100k_labels_images_{}.json'.format(flag))
    BDD_labels = json.load(open(BDD_labels_dir, 'r'))
    BDD_images_dir = os.path.join(BDD_path, 'images/100k/{}'.format(flag))

    for cnt, bdd in enumerate(BDD_labels):
        img_name = bdd['name']
        img_path = os.path.join(BDD_images_dir, img_name)
        # img = cv2.imread(img_path)
        # h, w, d = img.shape
        h, w, d = 720, 1280, 3
        bdd_boxes = bdd['labels']
        box_list, label_list = [], []
        for bb in bdd_boxes:
            if bb['category'] not in label_map.keys():
                continue
            box = bb['box2d']
            box_list.append([round(box['x1']), round(box['y1']),
                             round(box['x2']), round(box['y2'])])
            label_list.append(bb['category'])

        if len(box_list) != 0:
            save_path = os.path.join('/data/BDD100K/BDD100K_VOC/bdd100k_{}/Annotations'.format(flag),
                                     img_name.replace('.jpg', '.xml'))
            write_xml(save_path, img_name, box_list, label_list, w, h, d)
        if cnt % 100 == 0:
            print('{} process: {}/{}'.format(flag, cnt+1, len(BDD_labels)))
    print('Finish!')









