import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from utils.der_gssp_buffer import Buffer
from utils.derpp_buffer import Buffer as Buffer_RSVR

import matplotlib.pyplot as plt

def get_parser(parser):
    return parser



class Derppgssrev(nn.Module):
    NAME = 'derppgssrev'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(Derppgssrev, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = Buffer(self.args.buffer_size, self.device, minibatch_size=8, model_name=self.NAME,model=self)
        self.buffer_r = Buffer_RSVR(self.args.buffer_size, self.device, self.NAME)

    #gss function get_grads
    def get_grads(self, inputs, labels):
        self.net.eval()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        grads = self.net.get_grads().clone().detach()
        self.opt.zero_grad()
        self.net.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads


    def observe(self, inputs, labels, task_id=None, record_list=None):

        self.buffer.drop_cache()
        self.buffer.reset_fathom()


        self.opt.zero_grad()
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs

        #Drawing the heatmap pictures:
        '''
        log_lanescore, heatmap, heatmap_reg = outputs
        ls, y = labels
        heatmap_np = heatmap.cpu().detach().numpy()

        for i in range(heatmap_np.shape[0]):
            plt.imshow(heatmap_np[i], cmap='hot',interpolation='nearest')
            # plt.colorbar()
            plt.title(f"Heatmap{i}")
            # plt.show()
            plt.savefig(f'/home/jacklin/Pictures/Heatmap_demo_0{i}.png')
        '''
        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
        loss = self.loss(outputs_prediction, labels) # OverallLoss

   
        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, give_index=False)
            buf_outputs = self.net(buf_inputs) 
            buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
            
            loss += self.args.alpha*F.mse_loss(buf_heatmap_logits, buf_logits) # heatmaps are logits 
            # print("gss logits loss")
  
            del buf_inputs, buf_logits, buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
            torch.cuda.empty_cache()

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, give_index=False)
            buf_outputs = self.net(buf_inputs)
            loss += self.loss(buf_outputs, buf_labels) 
            # print("gss loss++")
        
        if not self.buffer_r.is_empty():
            buf_inputs, _, buf_logits = self.buffer_r.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
            loss += self.args.alpha * F.mse_loss(buf_heatmap_logits, buf_logits)
            # print("rsvr logits loss")
            
            del buf_inputs, buf_logits, buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
            torch.cuda.empty_cache()

            buf_inputs, buf_labels, _ = self.buffer_r.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            #set the beta as 1
            loss +=  self.loss(buf_outputs, buf_labels)
            # print("rsvr loss++")



        loss.backward()
        self.opt.step()
        
        if not self.buffer.is_empty:
            del buf_inputs, buf_outputs, buf_labels
            torch.cuda.empty_cache()

        if task_id is not None and record_list is not None:
            self.buffer.add_data(examples=inputs, labels=labels, logits=heatmap.detach(), task_order=task_id, record_data_list=record_list)
            self.buffer_r.add_data(examples=inputs, labels=labels, logits=heatmap.detach(), task_order=task_id, record_data_list=record_list)
        else:
            self.buffer.add_data(examples=inputs, labels=labels, logits=heatmap.detach())
            self.buffer_r.add_data(examples=inputs, labels=labels, logits=heatmap.detach())
        return loss.item()

