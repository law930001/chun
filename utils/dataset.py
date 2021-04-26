from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

from utils import order_points_clockwise, image_label

# return dataset
def get_dataset(config):
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    train_data_path = config['train_data_path']
    train_data_list = get_datalist(train_data_path)

    dataset = ImageDataset(transform=train_transform, 
                            data_list=train_data_list,
                            input_size=config['train_args']['input_size'],
                            img_channel=config['train_args']['img_channel'],
                            shrink_ratio=config['train_args']['shrink_ratio'])
    return dataset

# return dataset
def get_datalist(train_data_path):
    
    train_data_list= []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
            img_path = line[0]
            label_path = line[1]
            train_data_list.append((str(img_path), str(label_path)))
    
    return train_data_list


class ImageDataset(Dataset):
    def __init__(self, data_list: list, input_size: int, img_channel: int, shrink_ratio: float, transform=None,
                 target_transform=None):
        self.data_list = self.load_data(data_list)
        self.input_size = input_size
        self.img_channel = img_channel
        self.transform = transform
        self.target_transform = target_transform
        self.shrink_ratio = shrink_ratio

    def __getitem__(self, index):
        img_path, text_polys, text_tags = self.data_list[index]
        im = cv2.imread(img_path, 1 if self.img_channel == 3 else 0)
        if self.img_channel == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img, score_map, training_mask = image_label(im, text_polys, text_tags, self.input_size,
                                                    self.shrink_ratio)
        # score_map=score_map*250.0
        # score_map = score_map.astype(np.uint8)
        # training_mask=training_mask*250.0
        # training_mask = training_mask.astype(np.uint8)
        # print(score_map)
        # cv2.imwrite('./img.jpg', img)
        # cv2.imwrite('./score_map1.jpg', score_map[0])
        # cv2.imwrite('./score_map2.jpg', score_map[1])
        # cv2.imwrite('./training_mask.jpg', training_mask)

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_map = self.target_transform(score_map)
            training_mask = self.target_transform(training_mask)
        return img, score_map, training_mask

    def load_data(self, data_list: list) -> list:
        t_data_list = []
        for img_path, label_path in data_list:
            bboxs, text_tags = self._get_annotation(label_path)
            if len(bboxs) > 0:
                t_data_list.append((img_path, bboxs, text_tags))
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> tuple:
        boxes = []
        text_tags = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.arcLength(box, True) > 0:
                        boxes.append(box)
                        label = params[8]
                        if label == '*' or label == '###':
                            text_tags.append(False)
                        else:
                            text_tags.append(True)
                except:
                    print('load label failed on {}'.format(label_path))
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)