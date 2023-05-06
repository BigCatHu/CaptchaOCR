import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import my_datasets
from model import mymodel


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    train_datas=my_datasets.mydatasets(r"./dataset/train/")
    test_data=my_datasets.mydatasets(r"./dataset/test/")
    train_dataloader=DataLoader(train_datas,batch_size=64,shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    writer=SummaryWriter("logs")
    m=mymodel().to(device)

    loss_fn=nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    w=SummaryWriter("logs")
    total_step=0

for i in range(2):
    for i,(imgs,targets) in enumerate(train_dataloader):
        imgs=imgs.to(device)
        targets=targets.to(device)
        # print(imgs.shape)
        # print(targets.shape)
        outputs=m(imgs)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if i%10==0:
            total_step+=1
            print("训练{}次,loss:{}".format(total_step*10, loss.item()))
            w.add_scalar("loss",loss,total_step)

        # writer.add_images("imgs", imgs, i)
    writer.close()

torch.save(m.state_dict(),"model.pth")
