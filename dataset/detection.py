import os
import glob
import numpy as np
from lxml import etree
import torch.utils.data as data
from matplotlib import pyplot as plt
    
    
class AnnotationTransform(object):
    def __init__(self, class_map, valid_status=('3', '5')):
        self.class_map = class_map
        self.valid_status = valid_status
        
    def __call__(self, sample):
        image = sample['image']
        annotation = sample['annotation']
        size = (int(annotation['size']['width']), int(annotation['size']['height']))
        boxes = []
        polygons = []
        labels = []
        for i in annotation['object']:
            if 'status' in i:
                if i['status'] not in self.valid_status:
                    continue
                    
            if self.class_map is not None:
                labels.append(self.class_map[i['name']][0])
            else:
                labels.append(i['name'])
                
            if 'bndbox' in i:
                boxes.append([[float(i['bndbox']['xmin']) / size[0], float(i['bndbox']['ymin']) / size[1]], 
                              [float(i['bndbox']['xmax']) / size[0], float(i['bndbox']['ymax']) / size[1]]])
            elif 'polygon' in i:
                polygons.append(np.array([[float(j['x']) / size[0], float(j['y']) / size[1]] 
                                          for j in i['polygon']['point']]))
            else:
                raise Exception('\{} Not Supported!'.format(i))
                
        boxes = np.array(boxes, dtype=np.float32)
        polygons = np.array(polygons)
        labels = np.array(labels)
        sample = {'image': image, 'boxes': boxes, 'polygons': polygons, 'labels': labels, 'class_map': self.class_map}
        return sample
    
    
class DetectionDataset(data.Dataset):
    def __init__(self, image_path, annotation_path, class_map, transform=None, with_path=False):
        image_paths = glob.glob('{}/*.jpg'.format(image_path))
        annotation_paths = glob.glob('{}/*.xml'.format(annotation_path))
        image_names = set([i.split('/')[-1].split('.')[0] for i in image_paths])
        annotation_names = set([i.split('/')[-1].split('.')[0] for i in annotation_paths])
        names = image_names & annotation_names
        names = list(sorted(names))
        self.image_paths = [os.path.join(image_path, '{}.jpg'.format(i)) for i in names]
        self.annotation_paths = [os.path.join(annotation_path, '{}.xml'.format(i)) for i in names]
        
        assert(len(self.image_paths) == len(self.annotation_paths))
        self.class_map = class_map
            
        self.with_path = with_path
        self.transform = transform
        self.target_transform = AnnotationTransform(class_map)
                
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        image = plt.imread(image_path)
        with open(annotation_path, 'rb') as fp:
            xml_str = fp.read()
            xml = etree.fromstring(xml_str)
            annoatation = self._recursive_parse_xml_to_dict(xml)["annotation"]
        if self.with_path:
            sample = {'image_path': image_path, 'annotation_path': annotation_path, 
                      'image': image, 'annotation': annoatation, 'class_map': self.class_map}
        else:
            sample = {'image': image, 'annotation': annoatation, 'class_map': self.class_map}
        sample = self.target_transform(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.image_paths)
    
    def _recursive_parse_xml_to_dict(self, xml):
        """Recursively parses XML contents to python dict.

        We assume that `object` tags are the only ones that can appear
        multiple times at the same level of a tree.

        Args:
          xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
          Python dictionary holding XML contents.
        """
        if xml is None:
            return {}
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self._recursive_parse_xml_to_dict(child)
            if child.tag not in {'object', 'point'}:
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}