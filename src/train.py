import torch
import os,sys
import argparse
from model import Resnet

def main():
    net = Resnet()
    opt = torch.optim.SGD(net.parameters())

    