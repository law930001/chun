import gc
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


from utils import load_json, get_dataset
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def train(config, model, train_dataloader):

    if torch.cuda.is_available():
        cudnn.benchmark = True
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # set optimizer
    if config['optimizer']['class'] == 'SGD':
        optimizer = torch.optim.SGD(model.get_parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'])
    else:
        optimizer = torch.optim.Adam(model.get_parameters(), lr=config['optimizer']['lr'])
    # set learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler']['step_size'], gamma=config['lr_scheduler']['gamma'])

    model.to_device(device)

    model.to_train()

    iter_num = model.iter

    if not os.path.isfile('./output/train_log'):
        open('./output/train_log', 'w')
    log_file = open('./output/train_log', 'a')

    print('Start Training')

    for epoch in range(model.epoch, 8000):
        for i, (images, labels, training_masks) in enumerate(train_dataloader):
            # img, score_map, training_mask
            images, labels, training_masks = images.to(device), labels.to(device), training_masks.to(device)

            pred = model(images)

            loss_all, loss_tex, loss_ker, loss_agg, loss_dis = model.criterion(pred, labels, training_masks)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            gc.collect()

            if i % 20 == 0:
                print('epoch: {}, total_iter: {}, loss_all: {}, lr: {}'.format(
                    epoch, iter_num, loss_all, optimizer.param_groups[0]['lr']))
                log_file.write('epoch: {}, total_iter: {}, loss_all: {}, lr: {}\n'.format(
                    epoch, iter_num, loss_all, optimizer.param_groups[0]['lr']))
                log_file.flush()

            if iter_num % 1000 == 0 and iter_num != model.iter:
                save_file_name = 'epoch_{}_iter_{}.pth'.format(epoch,iter_num)
                print('Saving {}'.format(save_file_name))
                torch.save(model.state_dict(), './output/{}'.format(save_file_name))
                log_file.write('Saving {}\n'.format(save_file_name))
                log_file.flush()

            iter_num += 1


        
        scheduler.step()



    print('train done')




def main(config):
    train_dataset = get_dataset(config)
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=config['train_args']['batch_size'],
                                shuffle=config['train_args']['shuffle'],
                                num_workers=config['train_args']['num_workers'])

    model = Model(config)

    train(config, model, train_dataloader)

    print('main done')



if __name__ == '__main__':

    config = load_json('./config/config_pan.json')

    main(config)