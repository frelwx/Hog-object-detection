import numbers
import torch
from tqdm import tqdm
from dataset import detection_dataset, random_flip
import torchvision
import time
from classifier import linear_classfier
best_acc = 0.0
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0

    for idx, data in enumerate(tqdm(loader), 0):
        feature, label = data[0].to(device).squeeze(), data[1].to(device).squeeze()
        predict = model(feature)
        loss = loss_fn(predict, label)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(idx % 200 == 199):
            print("train loss: ", running_loss)
            running_loss = 0.0

def test_one_epoch(model, loader, optimizer, loss_fn):
    global best_acc
    model.eval()
    running_loss = 0.0
    right = 0
    total = 0
    TP = []
    FP = []
    FN = []
    print("#" * 20 + "begin test!!!" )
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader), 0):
            feature, label = data[0].to(device).squeeze(), data[1].to(device).squeeze()
            predict = model(feature)
            loss = loss_fn(predict, label)
            running_loss += loss.item()

            predict = torch.argmax(predict, dim=-1)
            total += len(predict)
            right += torch.sum(predict == label).item()
            P = torch.nonzero(label == 0).squeeze()
            N = torch.nonzero(label == 1).squeeze()
            TP.append(torch.sum(predict[P] == label[P]).item())
            FP.append(torch.sum(predict[N] != label[N]).item())
            FN.append(torch.sum(predict[P] != label[P]).item())
            if(idx % 10 == 9):
                TP_2 = torch.as_tensor(TP)
                FP_2 = torch.as_tensor(FP)
                FN_2 = torch.as_tensor(FN)
                print(running_loss, right / total, torch.mean(TP_2 / (TP_2 + FP_2)), torch.mean(TP_2 / (TP_2 + FN_2)))
                running_loss = 0.0
                
    # if(right / total * 100 > best_acc):
    #     best_acc = right / total * 100
    #     torch.save(model.state_dict(), "./new_best2.pt")
    #     print("save!", best_acc)
    # print("right : ", right / total * 100)
if __name__ == "__main__":
    transforms = random_flip
    batch_size = 1
    loader = torch.utils.data.DataLoader(detection_dataset(), 
                        batch_size = batch_size, shuffle=True, num_workers=min(0 if batch_size == 1 else batch_size, 6))
    test_loader = torch.utils.data.DataLoader(detection_dataset(), 
                        batch_size = batch_size, shuffle=True, num_workers=min(0 if batch_size == 1 else batch_size, 6))
    device = torch.device("cuda:0")
    model = linear_classfier()
    model = model.to(device)
    # model.load_state_dict(torch.load("D:\LP\checkpoint\checkpoint_mobi3_1"))
    checkpoint_path = "/home/lwx/HOG/new_best2.pt"
    model.load_state_dict(torch.load(checkpoint_path))
    optimizer = torch.optim.SGD(params=model.parameters(), lr=2e-3, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    test_one_epoch(model=model, loader=test_loader, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss())
    # for epoch in range(0, 100):
    #     print("[Epoch:", epoch, "]")
    #     train_one_epoch(model=model, loader=loader, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss())
    #     if(epoch % 2 == 1):
    #         test_one_epoch(model=model, loader=test_loader, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss())
    #     # scheduler.step()
        
        
        