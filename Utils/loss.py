import torch
import math

# regression losses for the hurricane tracking :

class Regress_Loss(torch.nn.Module):

    def __init__(self):
        super(Regress_Loss,self).__init__()

    def forward(self,x,y):
        #example x: (300,2)
        #example y: (300,3,2)

        predicted = x + y[:,0,:]
        ground_true = y[:,1,:]
        # predicted : (300,2)
        # groud_true : (300,2)
        R=6371 # kms (earth's radius)
        phi_1 = predicted[:,1] / 180 * math.pi
        phi_2 = ground_true[:,1] / 180 * math.pi
        delta_phi = (ground_true[:,1] - predicted[:,1]) / 180 * math.pi
        delta_lambda = (ground_true[:,0] - predicted[:,0]) / 180 * math.pi
        a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi_1) * torch.cos(phi_2) * torch.pow(torch.sin(delta_lambda/2),2)
        a = a.float()
        c = 2*torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        c = R * c
        mean_loss = torch.sum(torch.pow(c,2))/x.shape[0]
        return mean_loss


class Regress_Loss_Mae(torch.nn.Module):

    def __init__(self):
        super(Regress_Loss_Mae,self).__init__()

    def forward(self,x,y):
        #example x: (300,2)
        #example y: (300,3,2)

        predicted = x + y[:,0,:]
        ground_true = y[:,1,:]
        # predicted : (300,2)
        # groud_true : (300,2)
        R=6371 # kms (earth's radius)
        phi_1 = predicted[:,1] / 180 * math.pi
        phi_2 = ground_true[:,1] / 180 * math.pi
        delta_phi = (ground_true[:,1] - predicted[:,1]) / 180 * math.pi
        delta_lambda = (ground_true[:,0] - predicted[:,0]) / 180 * math.pi
        a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi_1) * torch.cos(phi_2) * torch.pow(torch.sin(delta_lambda/2),2)
        a = a.float()
        c = 2*torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        c = R * c
        mean_loss = torch.sum(c)/x.shape[0]
        return mean_loss



