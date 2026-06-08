import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CatModel(nn.Module):
    def __init__(self, path, AEmodel, DAmodel):
        super().__init__()

        ae_checkpoint = torch.load(path)

        encoder_dict = {
        k: v for k, v in ae_checkpoint['model_state_dict'].items() 
        if k in AEmodel.encoder.state_dict() and v.shape == AEmodel.encoder.state_dict()[k].shape}

        decoder_dict = {
        k: v for k, v in ae_checkpoint['model_state_dict'].items() 
        if k in AEmodel.decoder.state_dict() and v.shape == AEmodel.decoder.state_dict()[k].shape}

        AEmodel.encoder.load_state_dict(encoder_dict)
        AEmodel.decoder.load_state_dict(decoder_dict)

        for param in AEmodel.encoder.parameters():
            param.requires_grad = False
        for param in AEmodel.decoder.parameters():
            param.requires_grad = False

        self.encoder = AEmodel.encoder
        self.intermediate = DAmodel
        self.decoder = AEmodel.decoder
        
        
    def forward(self, x, goes, station, x_time, geo, adj):
        x = torch.cat((x, geo), dim=1)
        # Get latent representation
        latent = self.encoder(x[:,:-2])
        
        # Process through intermediate model
        x = self.intermediate(latent, x[:,-7:], goes, station, x_time, adj)
        
        # Decode back
        x = self.decoder(x)

        b = x.shape[0]
        x = x.reshape((b, 16, 16, 16, 16, 24))
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape((b, 24, 256, 256))
        
        return x



class Cat4DModel(nn.Module):
    def __init__(self, path, AEmodel, DAmodel, f_path, YLNet):
        super().__init__()

        ae_checkpoint = torch.load(path)
        yl_checkpoint = torch.load(f_path, weights_only = True)
        

        encoder_dict = {
        k: v for k, v in ae_checkpoint['model_state_dict'].items() 
        if k in AEmodel.encoder.state_dict() and v.shape == AEmodel.encoder.state_dict()[k].shape}

        decoder_dict = {
        k: v for k, v in ae_checkpoint['model_state_dict'].items() 
        if k in AEmodel.decoder.state_dict() and v.shape == AEmodel.decoder.state_dict()[k].shape}

        AEmodel.encoder.load_state_dict(encoder_dict)
        AEmodel.decoder.load_state_dict(decoder_dict)
        YLNet.load_state_dict(yl_checkpoint['model_state_dict'])
        
        for param in AEmodel.encoder.parameters():
            param.requires_grad = False
            
        for param in AEmodel.decoder.parameters():
            param.requires_grad = False
            
        for param in YLNet.parameters():
            param.requires_grad = False
        
        self.encoder = AEmodel.encoder
        self.intermediate = DAmodel
        self.decoder = AEmodel.decoder
        self.fnet = YLNet
        
        
    def forward(self, x, goes, station, x_time, geo, adj,data_mean, data_std, yl_mean, yl_std):
        x = torch.cat((x, geo), dim=1)
        # Get latent representation
        latent = self.encoder(x[:,:-2])
        
        # Process through intermediate model
        x_i = self.intermediate(latent, x[:,-7:], goes, station, x_time[:,0], adj)
        
        # Decode back
        x = self.decoder(x_i)

        b = x.shape[0]
        x = x.reshape((b, 16, 16, 16, 16, 24))
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape((b, 24, 256, 256))
        x = (x * data_std + data_mean - yl_mean) / yl_std
        y = self.fnet(x, x_time, geo[:,0:1])        
        
        return y




class CatDAModel(nn.Module):
    def __init__(self, DAmodel, f_path, YLNet, num_timestamps):
        super().__init__()
        self.num_timestamps = num_timestamps

        yl_checkpoint = torch.load(f_path, weights_only = True)
        YLNet.load_state_dict(yl_checkpoint['model_state_dict'])
        for param in YLNet.parameters():
            param.requires_grad = False
        

        self.intermediate = DAmodel
        self.fnet = YLNet
        
    def spatial_gradient_builtin(self, x):

        grad_x = torch.gradient(x, dim=-1)[0]
        # 计算H方向的梯度 (dim=-2)  
        grad_y = torch.gradient(x, dim=-2)[0]
    
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x, goes, station, x_time, geo, adj):
        
        out = []
        Gradient = []
        
        xb = self.fnet(x, x_time[:,0], geo[:,0:1]).detach()
        in_grad = self.spatial_gradient_builtin(xb)
        in_grad = in_grad.detach()
        
        for i in range(self.num_timestamps):
            # Process through intermediate model
            x_a, grad = self.intermediate(xb, in_grad, geo, goes, station, x_time[:,i+1], adj)
            
            out.append(x_a)
            Gradient.append(grad)
        # y = self.fnet(x, x_time, geo[:,0:1])        
        
        return out, Gradient



class CatDAModel_eval(nn.Module):
    def __init__(self, DAmodel, f_path, YLNet, num_timestamps):
        super().__init__()
        self.num_timestamps = num_timestamps

        yl_checkpoint = torch.load(f_path, weights_only = True)
        YLNet.load_state_dict(yl_checkpoint['model_state_dict'])
        for param in YLNet.parameters():
            param.requires_grad = False
        

        self.intermediate = DAmodel
        self.fnet = YLNet
        
    def spatial_gradient_builtin(self, x):
        grad_x = torch.gradient(x, dim=-1)[0]
        grad_y = torch.gradient(x, dim=-2)[0]
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x, goes_data, station, x_time, geo, adj):
        
        xb, xg = self.fnet(x, x_time[:,0], geo[:,0:1])
        xb = xb.detach()
        xg = xg.detach()
        out = []
        Gradient = []
        background = []

        in_grad = self.spatial_gradient_builtin(xb)
        in_grad = in_grad.detach() 
        
        for i in range(0,self.num_timestamps):
            goes = goes_data[:,i:i+2].reshape((-1,12,440,408))
            # Process through intermediate model
            x_a, grad = self.intermediate(xb, in_grad, geo, goes, station[:,i], x_time[:,i+1], adj)
            out.append(x_a)
            Gradient.append(grad)
            background.append(xb)
            if i != self.num_timestamps - 1:
                xb, xg = self.fnet(x_a, x_time[:,i+1], geo[:,0:1])
                in_grad = self.spatial_gradient_builtin(xb)

        return out, background, Gradient
    


class CatDAModel_eval_marge(nn.Module):
    def __init__(self, DAmodel, f_path, YLNet, num_timestamps):
        super().__init__()
        self.num_timestamps = num_timestamps
        self.intermediate = DAmodel
        self.fnet = YLNet
        self.wm = torch.tensor(np.load('data_stats/ED_stat/wm69.npy'), dtype=torch.float32)
        self.wn = torch.tensor(np.load('data_stats/ED_stat/wn69.npy'), dtype=torch.float32)

    def spatial_gradient_builtin(self, x):

        grad_x = torch.gradient(x, dim=-1)[0]
        grad_y = torch.gradient(x, dim=-2)[0]
    
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x, goes_data, station, x_time, geo, adj, marge):
        self.wm = self.wm.to(x.device)
        self.wn = self.wn.to(x.device)
        
        xb, xg = self.fnet(x, x_time[:,0], geo[:,0:1])
        xb = xb.detach()
        xg = xg.detach()
        
        in_grad = self.spatial_gradient_builtin(xb)
        in_grad = in_grad.detach()             
        
        goes = goes_data[:,0:2].reshape((-1,12,440,408))
        x_a, grad = self.intermediate(xb, in_grad, geo, goes, station[:,0], x_time[:,1], adj)
        Gradient = [grad]
        out = [x_a]
        bg = [xb]

        for i in range(0,self.num_timestamps):
            x_a, xg = self.fnet(x_a, x_time[:,i+1], geo[:,0:1])
            xb , xg = self.fnet(xb, x_time[:,i+1], geo[:,0:1])
            out.append(x_a)
            bg.append(xb)
            Gradient.append(xg)
            
            # x_a = x_a * self.wm + marge[:,i] * self.wn
            # xb = xb * self.wm + marge[:,i] * self.wn

        return out, bg, Gradient
    
    
class CatDAModel_eval_cycle_marge(nn.Module):
    def __init__(self, DAmodel, f_path, YLNet, num_timestamps):
        super().__init__()
        self.num_timestamps = num_timestamps
        self.intermediate = DAmodel
        self.fnet = YLNet
        self.wm = torch.tensor(np.load('data_stats/ED_stat/wm69.npy'), dtype=torch.float32)
        self.wn = torch.tensor(np.load('data_stats/ED_stat/wn69.npy'), dtype=torch.float32)
        self.indices = [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18]
    def spatial_gradient_builtin(self, x):

        grad_x = torch.gradient(x, dim=-1)[0]
        grad_y = torch.gradient(x, dim=-2)[0]
    
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x, goes_data, station, x_time, geo, adj, marge):
        self.wm = self.wm.to(x.device)
        self.wn = self.wn.to(x.device)
        
        xb, xg = self.fnet(x, x_time[:,0], geo[:,0:1])
        xb = xb.detach()
        xg = xg.detach()
        out = []
        Gradient = []
        background = [xb]
        xb = xb * self.wm + marge[:,0] * self.wn
        
        # xb[:, self.indices]  = xb[:, self.indices]  * self.wm + marge[:,0][:, self.indices]  * self.wn
        in_grad = self.spatial_gradient_builtin(xb)
        in_grad = in_grad.detach() 
        
        for i in range(0,self.num_timestamps):
            # goes = goes_data[:,i:i+2].reshape((-1,12,440,408))
            # # Process through intermediate model
            # x_a, grad = self.intermediate(xb, in_grad, geo, goes, station[:,i], x_time[:,i+1], adj)
            x_a = marge[:,i]
            out.append(x_a)
            Gradient.append(in_grad)
            
            if i != self.num_timestamps - 1:
                xb, xg = self.fnet(xb, x_time[:,i+1], geo[:,0:1])
                background.append(xb)
                xb = xb * self.wm + marge[:,i+1] * self.wn
                # xb[:, self.indices]  = xb[:, self.indices]  * self.wm + marge[:,i+1][:, self.indices]  * self.wn
                in_grad = self.spatial_gradient_builtin(xb)

        return out, background, Gradient
    

class CatDAModel_eval_cycle_hr(nn.Module):
    def __init__(self, DAmodel, f_path, YLNet, num_timestamps):
        super().__init__()
        self.num_timestamps = num_timestamps
        self.intermediate = DAmodel
        self.fnet = YLNet
        self.wm = torch.tensor(np.load('data_stats/ED_stat/wm69.npy'), dtype=torch.float32)
        self.wn = torch.tensor(np.load('data_stats/ED_stat/wn69.npy'), dtype=torch.float32)

    def spatial_gradient_builtin(self, x):

        grad_x = torch.gradient(x, dim=-1)[0]
        grad_y = torch.gradient(x, dim=-2)[0]
    
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x, goes_data, station, x_time, geo, adj, marge):
        self.wm = self.wm.to(x.device)
        self.wn = self.wn.to(x.device)

        out = []
        Gradient = []
        background = []
        x_a = x
        for i in range(0, self.num_timestamps):     

            xb, xg = self.fnet(x_a, x_time[:,i], geo[:,0:1])
            background.append(xb) 
            
                     

            # Process through intermediate model
            if (i+1) % 6 == 0: 
                xb = xb * self.wm + marge[:,i] * self.wn
                in_grad = self.spatial_gradient_builtin(xb)   
                goes = goes_data[:,i:i+2].reshape((-1,12,440,408))  
                x_a, grad = self.intermediate(xb, in_grad, geo, goes, station[:,i], x_time[:,i+1], adj)
                out.append(x_a)    
            else:
                out.append(xb)                 
                xb = xb * self.wm + marge[:,i] * self.wn
                in_grad = self.spatial_gradient_builtin(xb)   
                x_a = xb
   
                
            
            Gradient.append(in_grad)
           

        return out, background, Gradient


class CatDAModel_eval_lead_hr(nn.Module):
    def __init__(self, DAmodel, f_path, YLNet, num_timestamps):
        super().__init__()
        self.num_timestamps = num_timestamps
        self.intermediate = DAmodel
        self.fnet = YLNet
        self.wm = torch.tensor(np.load('data_stats/ED_stat/wm69.npy'), dtype=torch.float32)
        self.wn = torch.tensor(np.load('data_stats/ED_stat/wn69.npy'), dtype=torch.float32)

    def spatial_gradient_builtin(self, x):

        grad_x = torch.gradient(x, dim=-1)[0]
        grad_y = torch.gradient(x, dim=-2)[0]
    
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x, goes_data, station, x_time, geo, adj, marge):
        self.wm = self.wm.to(x.device)
        self.wn = self.wn.to(x.device)

        xb = x
        a = 6
        
        for i in range(0,a):     
            x, _ = self.fnet(x, x_time[:,i], geo[:,0:1])
        xb = x 
        in_grad = self.spatial_gradient_builtin(xb)
        goes = goes_data[:,0:2].reshape((-1,12,440,408))  
        x_a, _ = self.intermediate(xb, in_grad, geo, goes, station[:,0], x_time[:,a], adj)    
        background = [xb]
        Gradient = [in_grad]
        out = [x_a]
            
                

        for i in range(0, self.num_timestamps):
            xb, _ = self.fnet(xb, x_time[:,i+a], geo[:,0:1])
            out.append(xb)
            xb = xb * self.wm + marge[:,i] * self.wn
            Gradient.append(in_grad)
            background.append(xb)            

        return out, background, Gradient


class FixDAModel(nn.Module):
    def __init__(self, DAmodel, f_path, YLNet, Fixmodel):
        super().__init__()

        for param in YLNet.parameters():
             param.requires_grad = False
        for param in DAmodel.parameters():
             param.requires_grad = False       

        self.intermediate = DAmodel
        self.fnet = YLNet      
          
        self.fix = Fixmodel
        self.sigmoid = nn.Sigmoid()
        
    def spatial_gradient_builtin(self, x):

        grad_x = torch.gradient(x, dim=-1)[0]
        # 计算H方向的梯度 (dim=-2)  
        grad_y = torch.gradient(x, dim=-2)[0]
    
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x, goes_data, station, x_time, geo, adj, marge):
        
        xb,_ = self.fnet(x, x_time[:,0], geo[:,0:1])

        in_grad = self.spatial_gradient_builtin(xb)
        in_grad = in_grad.detach() 

        goes = goes_data[:,0:2].reshape((-1,12,440,408))
        x_a, _ = self.intermediate(xb, in_grad, geo, goes, station[:,0], x_time[:,1], adj)
        xb = self.fnet(x_a, x_time[:,1], geo[:,0:1]).detach()
        
        xb = torch.cat([xb, marge], dim=1)
        Wm = self.sigmoid(self.fix(xb, x_time[:,2], geo[:,0:1]))
        x = xb[:,:24] * Wm + xb[:,:24] * (1.0 - Wm)
        
        return x, xb[:,:24]