import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(CAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class mCAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n*5, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        
        dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y3 = _perceive_with(x,dlap)
        y4 = _perceive_with(x,dlap2)
        # y = torch.cat((x,y1,y2,y3),1)
        y = torch.cat((x,y1,y2,y3,y4),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class mCAModelv2(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModelv2, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        
        dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        # y3 = _perceive_with(x,dlap)
        # y4 = _perceive_with(x,dlap2)
        # y = torch.cat((x,y1,y2,y3),1)
        #y = torch.cat((x,y1,y2,y3,y4),1)
        y = torch.cat((x,y1,y2),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)

        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class mCAModel_v3(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModel_v3, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n*6, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
#             return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
            temp = F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
            return  F.relu(temp)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        
        dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        dlap3 = np.array([[-1, -1, -1], [1, 8, -1],[-1,-1,-1]]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y3 = _perceive_with(x,dlap)
        y4 = _perceive_with(x,dlap2)
        y5 = _perceive_with(x,dlap3)
        # y = torch.cat((x,y1,y2,y3),1)
        #y = torch.cat((x,y1,y2,y3,y4),1)
        y = torch.cat((x,y1,y2,y3,y4,y5),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        
        dx = self.dropout1(dx)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.dropout2(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class mCAModel_v4(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModel_v4, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n*6, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
#             temp = F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
#             return  F.relu(temp)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        
        dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        dlap3 = np.array([[-1, -1, -1], [1, 8, -1],[-1,-1,-1]]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y3 = _perceive_with(x,dlap)
        y4 = _perceive_with(x,dlap2)
        y5 = _perceive_with(x,dlap3)
        # y = torch.cat((x,y1,y2,y3),1)
        #y = torch.cat((x,y1,y2,y3,y4),1)
        y = torch.cat((x,y1,y2,y3,y4,y5),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        
        dx = self.dropout1(dx)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.dropout2(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class mCAModel_v5(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModel_v5, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n*6, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
#         self.dropout1 = nn.Dropout(0.2)
#         self.dropout2 = nn.Dropout(0.5)

        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
#             return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
            temp = F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
            return  F.relu(temp)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        
        dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        dlap3 = np.array([[-1, -1, -1], [1, 8, -1],[-1,-1,-1]]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y3 = _perceive_with(x,dlap)
        y4 = _perceive_with(x,dlap2)
        y5 = _perceive_with(x,dlap3)
        # y = torch.cat((x,y1,y2,y3),1)
        #y = torch.cat((x,y1,y2,y3,y4),1)
        y = torch.cat((x,y1,y2,y3,y4,y5),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        
#         dx = self.dropout1(dx)
        dx = self.fc0(dx)
        dx = F.relu(dx)
#         dx = self.dropout2(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class mCAModel_v6(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModel_v6, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n*5, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        #dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dx = np.outer([1, 2, 1], [-1, 0, 1])   # Sobel filter
        
        #dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]])   # Sobel filter
        #dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]])   # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y3 = _perceive_with(x,dlap)
        y4 = _perceive_with(x,dlap2)
        # y = torch.cat((x,y1,y2,y3),1)
        y = torch.cat((x,y1,y2,y3,y4),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x

class mCAModel_v7(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModel_v7, self).__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        # self.fc0 = nn.Linear(channel_n*5, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        #dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dx = np.outer([1, 2, 1], [-1, 0, 1])   # Sobel filter
        
        #dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        #dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]])   # Sobel filter
        #dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        #dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]])   # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        # y3 = _perceive_with(x,dlap)
        # y4 = _perceive_with(x,dlap2)
        y = torch.cat((x,y1,y2),1)
        # y = torch.cat((x,y1,y2,y3,y4),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x

class mCAModel_v8(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(mCAModel_v8, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n*5, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(channel_n*5, channel_n, bias=False)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        #dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dx = np.outer([1, 2, 1], [-1, 0, 1])   # Sobel filter
        
        #dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]]) / 8.0  # Sobel filter
        dlap = np.array([[1, 2, 1], [2, -12, 2],[1,2,1]])   # Sobel filter
        #dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]]) / 8.0  # Sobel filter
        dlap2 = np.array([[0, 1, 0], [1, -4, 1],[0,1,0]])   # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y3 = _perceive_with(x,dlap)
        y4 = _perceive_with(x,dlap2)
        # y = torch.cat((x,y1,y2,y3),1)
        y = torch.cat((x,y1,y2,y3,y4),1)
        #y = torch.cat((x,y1,y2,y3,torch.abs(x), torch.abs(y1),torch.abs(y2), torch.abs(y3)),1)
        # yx = torch.cat(y,torch.abs(y),1)
        # yx = y
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.dropout1(dx)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.dropout2(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class smallModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(smallModel, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(channel_n, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        #y = torch.cat((x,y1,y2),1)
        y = x
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    


    
class LCAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(LCAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        # self.fc0 = nn.Linear(channel_n, hidden_size)
        # self.fc0 = nn.Linear(channel_n*3, channel_n)
        #self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        self.fc1 = nn.Linear(channel_n*3, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y
        # return x

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        #dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    

class CCAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, num_filters = 16, kernel_size=3):
        super(CCAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.num_filters = num_filters

        self.conv1 = nn.Conv2d(in_channels=channel_n, out_channels=self.num_filters, kernel_size=3, stride=1, padding=1)
        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc0 = nn.Linear(self.num_filters, hidden_size)
        # self.fc0 = nn.Linear(channel_n*3, channel_n)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            #return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

    #         dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
    #         dy = dx.T
    #         c = np.cos(angle*np.pi/180)
    #         s = np.sin(angle*np.pi/180)
    #         w1 = c*dx-s*dy
    #         w2 = s*dx+c*dy

    #         y1 = _perceive_with(x, w1)
    #         y2 = _perceive_with(x, w2)
    #         y = torch.cat((x,y1,y2),1)
        y = self.conv1(self.channel_n, 16, 3, 1, 1)
        return y
        # return x

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        #dx = self.perceive(x, angle)
        dx = self.conv1(x)
        dx = F.relu(dx)
        
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class CCAModel2(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, num_filters = 16, kernel_size=3):
        super(CCAModel2, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.num_filters = num_filters

        self.conv1 = nn.Conv2d(in_channels=channel_n, out_channels=self.num_filters, kernel_size=3, stride=1, padding=1)
        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        # self.fc0 = nn.Linear(self.num_filters, hidden_size)
        # self.fc0 = nn.Linear(channel_n*3, channel_n)
        #self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        self.fc1 = nn.Linear(self.num_filters, channel_n, bias = False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            #return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

    #         dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
    #         dy = dx.T
    #         c = np.cos(angle*np.pi/180)
    #         s = np.sin(angle*np.pi/180)
    #         w1 = c*dx-s*dy
    #         w2 = s*dx+c*dy

    #         y1 = _perceive_with(x, w1)
    #         y2 = _perceive_with(x, w2)
    #         y = torch.cat((x,y1,y2),1)
        y = self.conv1(self.channel_n, 16, 3, 1, 1)
        return y
        # return x

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        #dx = self.perceive(x, angle)
        dx = self.conv1(x)
        dx = F.relu(dx)
        
        dx = dx.transpose(1,3)
        # dx = self.fc0(dx)
        # dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class StudentModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(StudentModel, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        #self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        self.fc1 = nn.Linear(channel_n*3, channel_n, bias=False)
        # self.fc1 = nn.Linear(channel_n*3, channel_n)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        #dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
    
class SAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, k_filters=128):
        super(SAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n

        #self.fc0 = nn.Linear(channel_n*3, hidden_size)
        #self.fc0 = nn.Linear(channel_n, hidden_size)
        # self.fc0 = nn.Linear(16, hidden_size)
        #self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc1 = nn.Linear(128, channel_n, bias=False)
        self.fc1 = nn.Linear(k_filters, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
            
        #self.wp = torch.nn.Conv2d(channel_n, 128, 3, stride=1, padding=1, groups=self.channel_n)
        #self.wp = torch.nn.Conv2d(channel_n, 100, 3, stride=1, padding=1, groups=self.channel_n)
        self.wp = torch.nn.Conv2d(channel_n, k_filters, 3, stride=1, padding=1, groups=self.channel_n)
        #self.w1 = torch.nn.Conv2d(128, 128, 1, groups=self.channel_n)
        #self.w1 = torch.nn.Conv2d(128, 8, 1, groups=8)
        #self.w2 = torch.nn.Conv2d(128, channel_n, 1, bias=False, groups=self.channel_n)
        #self.w2 = torch.nn.Conv2d(128, 8, 1, bias=False, groups=8)
        #self.w2.weight.data.zero_()
            
        

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1
    
    def get_living_mask(self, x): # Cell is alive IF any neighbors have alpha>0.1
        colors = x[:, 3:4, :, :] # Batch, channel, h, w
        pooled = torch.nn.functional.max_pool2d(colors, 3, 1, 1)
        summed = torch.sum(pooled, dim=1)
        return summed > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        #import pdb; pdb.set_trace()
        pre_life_mask = self.alive(x)

        # dx = self.perceive(x, angle)
        # dx = dx.transpose(1,3)
        # dx = self.fc0(dx)
        # dx = F.relu(dx)
        # dx = self.fc1(dx)
        
        
        
        y = self.wp(x)
        # y = F.tanh(y)
        y = F.relu(y)
        #y = self.w1(y)
        #y = F.relu(y)
        # y = F.tanh(y)
        # y = self.w2(y)
        #y = F.relu(y)
        dx = y
        # x_new = x + y
        
        dx = dx.transpose(1,3)
        # dx = self.fc0(dx)
        # dx = F.relu(dx)
        dx = self.fc1(dx)
        
        
        # import pdb; pdb.set_trace()

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        # x = x+dx
        
        # post_life_mask = self.get_living_mask(x_new)
        # life_mask = pre_life_mask & post_life_mask
        # life_mask = life_mask[:,None,:,:]
        # x_new *= life_mask

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)
        
        # return x
        
        

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x