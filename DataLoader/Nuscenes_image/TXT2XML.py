#############################################################
# #### txt to xml
#############################################################
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from PIL import Image
import cv2


dict = {
    '0': 'pedestrian',
    '1': 'bicycle',
    '2': 'motorcycle',
    '3': 'car',
    '4': 'bus',
    '5': 'truck',
    '6': 'emergency',
    '7': 'vehicle.construction',
    '8': 'vehicle.trailer',
    '9': 'animal'
}


class Xml_make(object):
    def __init__(self):
        super().__init__()

    def __indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.__indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _imageinfo(self, list_top):
        annotation_root = ET.Element('annotation')
        annotation_root.set('verified', 'no')
        tree = ET.ElementTree(annotation_root)
        '''
        0:xml_savepath 1:folder,2:filename,3:path
        4:checked,5:width,6:height,7:depth
        '''
        folder_element = ET.Element('folder')
        folder_element.text = list_top[1]
        annotation_root.append(folder_element)

        filename_element = ET.Element('filename')
        filename_element.text = list_top[2]
        annotation_root.append(filename_element)

        path_element = ET.Element('path')
        path_element.text = list_top[3]
        annotation_root.append(path_element)

        checked_element = ET.Element('checked')
        checked_element.text = list_top[4]
        annotation_root.append(checked_element)

        source_element = ET.Element('source')
        database_element = SubElement(source_element, 'database')
        database_element.text = 'Unknown'
        annotation_root.append(source_element)

        size_element = ET.Element('size')
        width_element = SubElement(size_element, 'width')
        width_element.text = str(list_top[5])
        height_element = SubElement(size_element, 'height')
        height_element.text = str(list_top[6])
        depth_element = SubElement(size_element, 'depth')
        depth_element.text = str(list_top[7])
        annotation_root.append(size_element)

        segmented_person_element = ET.Element('segmented')
        segmented_person_element.text = '0'
        annotation_root.append(segmented_person_element)

        return tree, annotation_root

    def _bndbox(self, annotation_root, list_bndbox):
        for i in range(0, len(list_bndbox), 8):
            object_element = ET.Element('object')
            name_element = SubElement(object_element, 'name')
            name_element.text = list_bndbox[i]

            # flag_element = SubElement(object_element, 'flag')
            # flag_element.text = list_bndbox[i + 1]

            pose_element = SubElement(object_element, 'pose')
            pose_element.text = list_bndbox[i + 1]

            truncated_element = SubElement(object_element, 'truncated')
            truncated_element.text = list_bndbox[i + 2]

            difficult_element = SubElement(object_element, 'difficult')
            difficult_element.text = list_bndbox[i + 3]

            bndbox_element = SubElement(object_element, 'bndbox')
            xmin_element = SubElement(bndbox_element, 'xmin')
            xmin_element.text = str(list_bndbox[i + 4])

            ymin_element = SubElement(bndbox_element, 'ymin')
            ymin_element.text = str(list_bndbox[i + 5])

            xmax_element = SubElement(bndbox_element, 'xmax')
            xmax_element.text = str(list_bndbox[i + 6])

            ymax_element = SubElement(bndbox_element, 'ymax')
            ymax_element.text = str(list_bndbox[i + 7])

            annotation_root.append(object_element)

        return annotation_root

    def txt_to_xml(self, list_top, list_bndbox):
        tree, annotation_root = self._imageinfo(list_top)
        annotation_root = self._bndbox(annotation_root, list_bndbox)
        self.__indent(annotation_root)
        tree.write(list_top[0], encoding='utf-8', xml_declaration=True)


def txt_2_xml(source_path, xml_save_dir, txt_dir):
    COUNT = 0
    for folder_path_tuple, folder_name_list, file_name_list in os.walk(source_path):
        for file_name in file_name_list:
            file_suffix = os.path.splitext(file_name)[-1]
            if file_suffix != '.jpg':
                continue
            list_top = []
            list_bndbox = []
            path = os.path.join(folder_path_tuple, file_name)
            xml_save_path = os.path.join(xml_save_dir, file_name.replace(file_suffix, '.xml'))
            txt_path = os.path.join(txt_dir, file_name.replace(file_suffix, '.txt'))
            filename = os.path.splitext(file_name)[0]
            checked = 'NO'
            img = cv2.imread(path)
            height, width, depth = img.shape
            # print(height, width, depth)
            # im = Image.open(path)
            # im_w = im.size[0]
            # im_h = im.size[1]
            # width = str(im_w)
            # height = str(im_h)
            # depth = '3'
            flag = 'rectangle'
            pose = 'Unspecified'
            truncated = '0'
            difficult = '0'
            list_top.extend([xml_save_path, folder_path_tuple, filename, path, checked,
                             width, height, depth])
            for line in open(txt_path, 'r'):
                line = line.strip()
                info = line.split(' ')
                name = dict[info[0]]
                x_cen = float(info[1]) * width
                y_cen = float(info[2]) * height
                w = float(info[3]) * width
                h = float(info[4]) * height
                xmin = int(x_cen - w / 2)
                ymin = int(y_cen - h / 2)
                xmax = int(x_cen + w / 2)
                ymax = int(y_cen + h / 2)
                list_bndbox.extend([name, pose, truncated, difficult,
                                    str(xmin), str(ymin), str(xmax), str(ymax)])
            Xml_make().txt_to_xml(list_top, list_bndbox)
            COUNT += 1
            print(COUNT, xml_save_path)


if __name__ == '__main__':
    source_path = r'/home/ZT/ZT_Project/yolov4/VOCdevkit/VOC2007/JPEGImages'  # txt标注文件所对应的的图片
    xml_save_dir = r'/home/ZT/ZT_Project/yolov4/VOCdevkit/VOC2007/Annotations'  # 转换为xml标注文件的保存路径
    txt_dir = r'/home/ZT/ZT_Project/yolov4/VOCdevkit/VOC2007/TXT'  # 需要转换的txt标注文件
    txt_2_xml(source_path, xml_save_dir, txt_dir)
