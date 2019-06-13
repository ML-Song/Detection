import glob
import numpy as np
from lxml import etree
import torch.utils.data as data
from matplotlib import pyplot as plt
    
    
class AnnotationTransform(object):
    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind
        
    def __call__(self, sample):
        image = sample['image']
        annotation = sample['annotation']
        segmented = int(annotation['segmented'])
        size = (int(annotation['size']['width']), int(annotation['size']['height']))
        if not segmented:
            boxes = np.array([(int(i['bndbox']['xmin']), 
                               int(i['bndbox']['ymin']), 
                               int(i['bndbox']['xmax']), 
                               int(i['bndbox']['ymax'])) 
                              for i in annotation['object']], dtype=np.float32)
        else:
            raise Exception('\nSegmented Not Supported!')
        names = np.array([i['name'] for i in annotation['object']])
        if self.class_to_ind is not None:
            names = np.array([self.class_to_ind[i] for i in names])
        status = np.array([int(i['status']) for i in annotation['object']])
        boxes[:, [0, 2]] /= size[0]
        boxes[:, [1, 3]] /= size[1]
        
        return image, boxes[(status == 3) | (status == 5)], names[(status == 3) | (status == 5)]
    
    
class DetectionDataset(data.Dataset):
    def __init__(self, image_path, annotation_path, transform=None, class_to_ind=None, with_path=False):
        self.transform = transform
        self.image_paths = glob.glob('{}/*.jpg'.format(image_path))
        self.annotation_paths = glob.glob('{}/*.xml'.format(annotation_path))
        assert(len(self.image_paths) == len(self.annotation_paths))
        self.with_path = with_path
        self.class_to_ind = class_to_ind
        self.target_transform = AnnotationTransform(class_to_ind)
                
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
                      'image': image, 'annotation': annoatation}
        else:
            sample = {'image': image, 'annotation': annoatation}
            
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
        if not len(xml):
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self._recursive_parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}