"""
SRAM modules
"""
import math
import numpy as np
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

def bit2cond(bitWeight, hrs, lrs):
    """
    Draft: replace the binary values to conductance measurement
    """
    level0 = torch.ones(bitWeight[bitWeight==0].size()).mul(hrs)
    level1 = torch.ones(bitWeight[bitWeight==1].size()).mul(lrs)
    level2 = torch.ones(bitWeight[bitWeight==-1].size()).mul(lrs).mul(-1.)

    bitWeight[bitWeight==0] = level0.cuda()
    bitWeight[bitWeight==1] = level1.cuda()
    bitWeight[bitWeight==-1] = level2.cuda()

    # bitWeight = bitWeight.clamp(0)
    return bitWeight

def program_noise_cond(weight_q, weight_b, hrs, lrs, sensitive_lv, rram_type='6K'):
    wb = torch.zeros_like(weight_b)
    weight_cond = bit2cond(weight_b, hrs, lrs)  # typical values

    for ii in range(len(weight_q.unique())):
        lv = ii - 7

        if len(weight_q.unique()) == 1:
            ii = int(weight_q.unique().item())

        idx_4b = weight_q.eq(lv)
        wb_ii = weight_cond[:, idx_4b]
        wbin_ii = weight_b[:, idx_4b]
        
        # noises
        noise = np.load(f"/home/wangxinh/Data/TorchInference_RRAM_new/save/interface/noSWIPE_25Times_raw/level{abs(lv)}_raw.npy") #6K no-SWIPE
        ref = np.load(f"/home/wangxinh/Data/TorchInference_RRAM_new/save/interface/noSWIPE_25Times_raw/level0_raw.npy")
        
        if rram_type == "9K":
            noise = noise * (1.11e-4/1.66e-4)
            ref = ref * (1.11e-4/1.66e-4)
        
        if rram_type == "6K":
            swipe = np.load(f"/home/wangxinh/Data/TorchInference_RRAM_new/save/interface/prob/09191845/level{abs(lv)}_raw.npy") # 6K-SWIPE
        elif rram_type == "9K":
            swipe = np.load(f"/home/wangxinh/Data/TorchInference_RRAM_new/save/interface/prob/10201845/level{abs(lv)}_raw.npy") # 9K-SWIPE  

        # clean the hrs reference
        max_ref = ref.max(axis=1)
        ref = ref[max_ref<5e-6, :]

        # sizes
        _, numel = wb_ii.size()
        
        bit_idx = np.arange(noise.shape[0])
        random_idx = np.random.choice(bit_idx, size=(numel))
        
        bit_random_noise = noise[random_idx, :].T
        swipe_random_noise = swipe[random_idx, :].T

        wb_cond = torch.from_numpy(bit_random_noise).float()
        swipe_cond = torch.from_numpy(swipe_random_noise).float()

        wb_cond = torch.flip(wb_cond, dims=[0])
        swipe_cond = torch.flip(swipe_cond, dims=[0])

        # reference idx
        ref_idx = np.arange(ref.shape[0])
        random_idx = np.random.choice(ref_idx, size=(numel))
        ref_noise = ref[random_idx, :].T

        ref_cond = torch.from_numpy(ref_noise).float()
        ref_cond = torch.flip(ref_cond, dims=[0])
        # import pdb;pdb.set_trace()
        if lv < 0:
            wb_cond = ref_cond - wb_cond
            swipe_cond = ref_cond - swipe_cond

        if not lv in sensitive_lv:
            wb[:, idx_4b] = swipe_cond.cuda()   # SWIPE scheme
        else:
            wb[:, idx_4b] = wb_cond.cuda()      # Non SWIPE scheme

        # if lv == -1:
        #     import pdb;pdb.set_trace()
        # print(1)

    return wb

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
                    wl_input=8, wl_weight=8, subArray=128, cellBit=1, ADCprecision=8, rram_type="6K", sensitive_lv=[0]):
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
        self.rram_type = rram_type
        assert rram_type in ["6K", "9K"], "rram cell type must be 6K or 9K"
        
        # quantization
        self.wbit = wl_weight
        self.abit = wl_input 
        self.weight_quant = WQ(wbit=wl_weight)
        self.act_quant = AQ(abit=wl_input, act_alpha=torch.tensor(10.0))

        # conductance
        self.hrs = 1e-6
        if rram_type == "6K":
            self.lrs = 1.66e-04
        elif rram_type == "9K":
            self.lrs = 1.11e-04
        
        self.nonideal_unit = self.lrs - self.hrs
        self.sensitive_lv = sensitive_lv

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
        
        # deploy to conductance
        # wqb_list = program_noise_cond(wq, wqb_list, hrs=self.hrs, lrs=self.lrs, sensitive_lv=self.sensitive_lv)
        
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

        # program
        wqb_list = program_noise_cond(wint, wqb_list, hrs=self.hrs, lrs=self.lrs, sensitive_lv=self.sensitive_lv, rram_type=self.rram_type)

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
                            maci = maci.div(self.nonideal_unit)

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
                                maci = maci.div(self.nonideal_unit)
                                macs = macs + maci * scaler
                                
                            total_macs = total_macs.add(macs)
                        scalerIN = 2**z
                        outputIN = outputIN + total_macs * scalerIN
                    output = output + outputIN / x_scale 
        output = output / w_scale
        return output