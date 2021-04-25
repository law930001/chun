import matplotlib.pyplot as plt
import torch
import os
import cv2
import time
from torchvision import transforms

from utils.util import draw_bbox, load_json
from model import Model
from post_processing import decode

class demo_Model(Model):

    def __init__(self, config):
        super().__init__(config)

    def demo(self, img_path, short_size: int = 640):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale_h = short_size / h
        scale_w = short_size / w
        # print(h,w,scale)
        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        self.to_device(device)
        self.to_eval()
        
        with torch.no_grad():
            torch.cuda.synchronize(device)
            start = time.time()
            preds = self.forward(tensor)[0]
            torch.cuda.synchronize(device)
            preds, boxes_list = decode(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            t = time.time() - start
        return preds, boxes_list, t



if __name__ == '__main__':

    config = load_json('config.json')

    model_path = './output/epoch_99_iter_24700.pth'

    img_id = 10

    dirs = os.listdir('/root/Storage/datasets/plate/motor/')

    # 初始化网络
    config['model']['pretrained'] = False
    model = demo_Model(config)
    model.load_model(model_path)

    for i, file in enumerate(dirs):

        img_path = '/root/Storage/datasets/plate/motor/{}'.format(file)

        preds, boxes_list, t = model.demo(img_path)

        # show_img(preds)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::], boxes_list)
        # show_img(img, color=True)
        # plt.show()
        cv2.imwrite('./result/{}'.format(file), img)
        # print(boxes_list, t)

        print('save file {}'.format(file))



