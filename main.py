import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import testLoader
import model.Net as Net

if __name__ == '__main__':
    valSetPath = 'motorbike_val/'
    trainSetPath='motorbike_train/'
    valSet = testLoader.MyDataset(valSetPath)
    trainSet=testLoader.MyDataset(trainSetPath)
    valLoader=torch.utils.data.DataLoader(valSet, batch_size=16, num_workers=2, shuffle=False)
    trainLoader=torch.utils.data.DataLoader(trainSet, batch_size=16, num_workers=2, shuffle=False)

    print("ok")
    model = Net.Net()
    model.cuda()
    cudnn.benchmark = True
    print(model)

    # update net
    lr = 1e-5
    loss_func = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(list(model.parameters())[:], lr=lr)

    for epoch in range(100):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        losstemp=0
        iii=0
        for ind, tensors in enumerate(trainLoader):
            iii+=1
            tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
            #target
            inp, reflec_targ, shading_targ,mask = tensors
            reflec_targ=reflec_targ*mask
            shading_targ=shading_targ*mask
            #prediction
            reflec_pred,shading_pred = model(inp)
            reflec_pred=reflec_pred*mask
            shading_pred=shading_pred*mask
            #loss
            reflec_loss = loss_func(reflec_pred, reflec_targ)
            shding_loss = loss_func(shading_pred, shading_targ)
            train_loss = reflec_loss + shding_loss
            losstemp=train_loss.data

            if iii%100==0:
                print('Its :',ind)
                print(losstemp)
            #optimizer
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        #  if epoch % 100 == 0:
        print('Train Loss: {:.6f}'.format(train_loss / (len(trainSet))))

        if (epoch + 1) % 10 == 0:
            sodir = 'model/_iter_{}.pth'.format(epoch)
            print('Model save {}'.format(sodir))
            torch.save(model.state_dict(), sodir)

        # adjust
        if (epoch + 1) % 100 == 0:
            lr = lr / 10
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #val
        for ind, tensors in enumerate(valLoader):
            tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
            #target
            inp, reflec_targ, shading_targ,mask = tensors
            reflec_targ = reflec_targ * mask
            shading_targ = shading_targ * mask
            #prediction
            reflec_pred, shading_pred = model(inp)
            reflec_pred = reflec_pred * mask
            shading_pred = shading_pred * mask
            #save_result
            aa = transforms.ToPILImage()(inp[0].data.cpu()).convert('RGB')
            aa.save('res/'+str(ind) + '.jpg')
            a = transforms.ToPILImage()(reflec_pred[0].data.cpu()).convert('RGB')
            a.save('res/'+str(ind) + 're.jpg')
            aaa = transforms.ToPILImage()(shading_pred[0].data.cpu()).convert('RGB')
            aaa.save('res/'+str(ind) + 'sh.jpg')