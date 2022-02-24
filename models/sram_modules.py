"""
SRAM modules
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .quant_modules import WQ, AQ, stats_quant

def decimal2binary(weight_q, bitWeight, cellBit):
    cellRange = 2**cellBit
    remainder_list = torch.Tensor([]).type_as(weight_q)
    
    for k in range(int(bitWeight/cellBit)):
        remainder = torch.fmod(weight_q, cellRange)
        remainder = remainder.unsqueeze(0)
        remainder_list = torch.cat((remainder_list, remainder), dim=0)
        weight_q = torch.round((weight_q-remainder.squeeze(0))/cellRange)
    return remainder_list

def flip_twos(wqb_list, wbit, cellbit, negw):
    ones = wqb_list.eq(1.)*negw
    zeros = wqb_list.eq(0.)*negw

    for k in range(int(wbit/cellbit)):
        wqb = wqb_list[k]
        
        pos = ones[k]
        zrs = zeros[k]
        
        if k == 0:
            wqb[pos] = 0
            wqb[zrs] = 1.
            pos_prev = pos
        else:
            wqb[pos*pos_prev] = 0.
            wqb[zrs*pos_prev] = 1.
            pos_prev = pos * pos_prev
        
        wqb[negw] = wqb[negw] * (-1)
        wqb_list[k] = wqb
    return wqb_list

class SRAMConv2d(nn.Conv2d):
    r"""
    NeuroSim-based RRAM inference with low precision weights and activations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                wl_input=8, wl_weight=8, subArray=128, cellBit=1, ADCprecision=5):
        super(SRAMConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # sw and hw params
        self.wl_input = wl_input
        self.wl_weight = wl_weight
        self.cellBit = cellBit
        self.ADCprecision = ADCprecision
        self.layer_idx = 0
        self.iter = 0
        self.init = True
        self.subArray = subArray
        
        # quantization
        self.wbit = wl_weight
        self.abit = wl_input 
        self.weight_quant = WQ(wbit=wl_weight)
        self.act_quant = AQ(abit=wl_input, act_alpha=torch.tensor(10.0))

    def _act_quant(self, input):
        act_alpha = self.act_quant.act_alpha 
        input = torch.where(input < act_alpha, input, act_alpha)

        with torch.no_grad():
            scale = (2**self.abit - 1) / act_alpha
        
        input_div = input.mul(scale)
        input_q = input_div.round()
        return input_q, scale

    def forward(self, input: Tensor) -> Tensor:        
        # quantization
        wq, w_scale = stats_quant(self.weight.data, nbit=self.wbit, dequantize=False)
        wint = wq.clone()

        # negative weights
        negw = wq.lt(0.)
        
        # change the weights to positive for now 
        wq[negw] = wq[negw].add(2 ** (self.wbit - 1))
        
        # decomposition (decimal to binary)
        wqb_list = decimal2binary(wq, bitWeight=self.wbit, cellBit=self.cellBit)
        
        # flip the sign bit (2's complement)
        wqb_list[-1, negw] = 1.
        
        # get the bit values for the negative part
        ones = wqb_list.eq(1.)*negw
        zeros = wqb_list.eq(0.)*negw

        # flip the bits for the computation
        wqb_list[ones] = 0.
        wqb_list[zeros] = 1.

        # convert the 2's complement representation to signed bit values
        wqb_list = flip_twos(wqb_list, self.wl_weight, self.cellBit, negw)

        # input quantization
        xq, x_scale = self._act_quant(input)
        cellRange = 2**self.cellBit

        # targeted output size
        odim = math.floor((xq.size(2) + 2*self.padding[0] - self.dilation[0] * (wq.size(2)-1)-1)/self.stride[0] + 1)
        output = torch.zeros((xq.size(0), wq.size(0), odim, odim)).cuda()
        for i in range(wq.size(2)):
            for j in range(wq.size(3)):
                # numSubArray = wq.shape[1] // self.subArray
                numSubArray = 0
                
                if numSubArray == 0:
                    mask = torch.zeros_like(wq)
                    mask[:,:,i,j] = 1
                    xq, x_scale = self._act_quant(input)
                    outputIN = torch.zeros_like(output)
                    xb_list = []
                    for z in range(int(self.abit)):
                        xb = torch.fmod(xq, 2)
                        xq = torch.round((xq-xb)/2)
                        macs = torch.zeros_like(output).cuda()

                        xb_list.append(xb)
                        for k in range(int(self.wbit/self.cellBit)):
                            wqb = wqb_list[k]
                            scaler = cellRange**k

                            # partial sum
                            outputPartial = F.conv2d(xb, wqb*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            maci = outputPartial
                            macs = macs + maci * scaler

                        scalerIN = 2**z
                        outputIN = outputIN + macs * scalerIN
                    output = output + outputIN / x_scale 
                else:
                    xq, x_scale = self._act_quant(input)
                    outputIN = torch.zeros_like(output)
                    for z in range(int(self.abit)):
                        xb = torch.fmod(xq, 2)
                        xq = torch.round((xq-xb)/2)
                        total_macs = torch.zeros_like(output)

                        for s in range(numSubArray):
                            mask = torch.zeros_like(wq)
                            mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                            macs = torch.zeros_like(output).cuda()

                            for k in range(int(self.wbit/self.cellBit)):
                                wqb = wqb_list[k]
                                
                                # partial sum
                                outputPartial = F.conv2d(xb, wqb*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                scaler = cellRange**k

                                maci = outputPartial
                                macs = macs + maci * scaler
                                
                            total_macs = total_macs.add(macs)
                        scalerIN = 2**z
                        outputIN = outputIN + total_macs * scalerIN
                    output = output + outputIN / x_scale 
        output = output / w_scale
        return output