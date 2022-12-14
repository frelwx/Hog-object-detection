import numbers
import torch
from tqdm import tqdm
from dataset import recognition_dataset, random_flip
import torchvision
import time
from classifier import linear_classfier
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0

    for idx, data in enumerate(tqdm(loader), 0):
        feature, label = data[0].to(device), data[1].to(device)
        predict = model(feature)
        loss = loss_fn(predict, label)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(idx % 200 == 199):
            print(running_loss)
            running_loss = 0.0

def test_one_epoch(model, loader, optimizer, loss_fn):
    model.eval()
    running_loss = 0.0
    right = 0
    total = 0
    print("#" * 20 + "begin test!!!" )
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader), 0):
            feature, label = data[0].to(device), data[1].to(device)
            predict = model(feature)
            loss = loss_fn(predict, label)
            running_loss += loss.item()

            predict = torch.argmax(predict, dim=-1)
            total += len(predict)
            right += torch.sum(predict == label).item()


            # if(idx % 10 == 9):
            #     print(running_loss, right / total)
            #     running_loss = 0.0
    print("right : ", right / total * 100)
if __name__ == "__main__":
    transforms = random_flip
    batch_size = 1
    loader = torch.utils.data.DataLoader(recognition_dataset(split_path="/home/lwx/HOG/data/csgo225labeled/train.txt", transforms=transforms), 
                        batch_size = batch_size, shuffle=True, num_workers=min(batch_size, 6))
    test_loader = torch.utils.data.DataLoader(recognition_dataset(split_path="/home/lwx/HOG/data/csgo225labeled/test.txt"), 
                        batch_size = batch_size, shuffle=True, num_workers=min(batch_size, 6))
    device = torch.device("cuda:0")
    model = linear_classfier()
    model = model.to(device)
    # model.load_state_dict(torch.load("D:\LP\checkpoint\checkpoint_mobi3_1"))
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    for epoch in range(0, 10):
        train_one_epoch(model=model, loader=loader, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss())
        test_one_epoch(model=model, loader=test_loader, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss())
        # scheduler.step()
        