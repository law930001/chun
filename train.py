import gc
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


from utils import load_json, get_dataset
from model import Model


def train(config, model, train_dataloader):

    if torch.cuda.is_available():
        cudnn.benchmark = True
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # set optimizer
    optimizer = torch.optim.SGD(model.get_parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'])
    # set learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler']['step_size'], gamma=config['lr_scheduler']['gamma'])

    model.to_device(device)

    model.to_train()

    iter_num = 0

    print('Start Training')

    for epoch in range(1, 100):
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
                print('epoch: {}, total_iter: {}, loss_all: {}'.format(epoch, iter_num, loss_all))

            if iter_num % 100 == 0:
                save_file_name = 'epoch_{}_iter_{}.pth'.format(epoch,iter_num)
                print('Saving {}'.format(save_file_name))
                torch.save(model.state_dict(), './output/{}'.format(save_file_name))

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