# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCL(nn.Module):
    def __init__(self,temperature=0.1):
        super(DCL, self).__init__()
        self.temp = temperature
    def forward(self,z):
        #print(z.shape)
        #155,11,32 
        #155 can be changed due to batch size
        #normalization
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        #print(z_ori.shape)
        #155,32
        z_trans = z[:, 1:]  # n,k-1, z
        #print(z_trans.shape)
        #155,10, 32
        batch_size, num_trans, z_dim = z.shape
        #print(batch_size)
        #155
        #print(num_trans)
        #11
        #print(z_dim)
        #32
        
        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        #print(sim_matrix.shape)
        #batch_size,11,32
        #masks는 True False로 구성된 행렬 생성-> 자기 자신과 같으면 False (1,1):False
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        #print(torch.ones_like(sim_matrix).to(z)) -> 155,11,11의 1로 채워진 행렬 생성 
        #print(torch.eye(num_trans).unsqueeze(0).to(z)) -> identity matrix(diaogonal 1) 생성
        #print(mask)
        #print(mask.shape)
        #batch_size,11,11
        #sim_matrix는 자기 자신(Fasle)인 값을 제외하고 return 
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        #print(sim_matrix.shape)
        #batch_size,11,10
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1
        #print(trans_matrix.shape)
        #batch_size,10
        
        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp) # n,k-1
        #print(pos_sim.shape)
        #batch_size, 10 
        K = num_trans - 1
        scale = 1 / np.abs(np.log(1.0 / K))
        loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale

        loss_n = loss_tensor.mean(1)
        loss_a = -torch.log(1-pos_sim/trans_matrix)*scale
        loss_a = loss_a.mean(1)

        return loss_n,loss_a

