#!/usr/bin/env python3
import torch
import torch.nn as nn
import skc_cuda
import math
import time

class SKCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, output, stride):
        ctx.stride = stride
        output = skc_cuda.forward(input, weights, output, stride)[0]
        ctx.save_for_backward(input, weights)
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weights = ctx.saved_tensors
        d_input = torch.zeros(input.size()).cuda()
        d_weights = torch.zeros(weights.size()).cuda()
        d_input, d_weights = skc_cuda.backward(d_output, input, weights, d_input, d_weights, ctx.stride)
        return d_input, d_weights, None, None, None

class SKC(torch.nn.Module):
    def __init__(self, input_channel, output_channel, input_unit_dim, output_unit_dim, stride):
        super(SKC, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.input_unit_dim = input_unit_dim
        self.output_unit_dim = output_unit_dim
        self.stride = stride
        self.widow_n = output_channel // output_unit_dim

        self.weights = torch.nn.Parameter(torch.randn(self.output_unit_dim, self.input_unit_dim))
        self.bias = torch.nn.Parameter(torch.ones(self.output_channel))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.size(0)
        h = input.size(2)
        w = input.size(3)

        self.output_tmp = torch.zeros((batch_size, self.output_channel, h, w)).cuda()
        # bias = self.bias.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).unsqueeze(3)
        skc_output = SKCFunction.apply(input, torch.stack([self.weights] * self.widow_n, dim=1).view(self.output_channel, self.input_unit_dim).cuda(), self.output_tmp, self.stride)
        
        # return slc_output + bias
        return skc_output