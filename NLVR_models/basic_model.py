import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.model
    
    def load_model(self, path, epoch):
        state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch))['state_dict']
        self.load_state_dict(state_dict)

    def save_model(self, path, epoch, acc, loss):
        torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, path+'{}_epoch_{}.pth'.format(self.name, epoch))

    def compute_loss(self, output, target, meta_target):
        pass

    def train_(self, image, target, meta_target ):
        self.optimizer.zero_grad()
        output = self(image)
        loss = self.compute_loss(output, target, meta_target)
        loss.backward()
        self.optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, target, meta_target ):
        with torch.no_grad():
            output = self(image)
        loss = self.compute_loss(output, target, meta_target)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, image, target, meta_target ):
        with torch.no_grad():
            output = self(image)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy
    
    def test_ra_pairs_(self, image, target, ra_pair_matrix):
        with torch.no_grad():
            output = self(image)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        ra_pair_matrix_correct = pred.eq(target.data).reshape(-1,1,1)*ra_pair_matrix
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy, ra_pair_matrix_correct.sum(dim=0).numpy()