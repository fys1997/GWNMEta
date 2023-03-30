import torch

import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='')
parser.add_argument('--data', type=str, default='data/BJ-Flow', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=1024, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, help='')
parser.add_argument('--plotheatmap', type=str, default='False', help='')
parser.add_argument('--onlyEMC', type=bool, default=False, help='只使用EC矩阵')
parser.add_argument('--onlyNMC', type=bool, default=False, help='只使用NMC矩阵')
parser.add_argument('--onlyADP', type=bool, default=False, help='只使用adp矩阵')
parser.add_argument('--hops', type=int, default=2, help='GCN hops')
parser.add_argument("--output_dir", type=str, default="data/plotArray", help="Output directory.")
parser.add_argument("--plotName", type=str, default="plot")
parser.add_argument("--oldGPU", type=str, default="cuda:2")

args = parser.parse_args()


def generate_x(min_t, max_t):
    f = h5py.File('data/BJ_FLOW.h5', 'r')
    data = np.array(f['/data'][()], np.float32)  # date hour x y 2
    days, hours, rows, cols, _ = data.shape
    data = np.reshape(data, (days * hours, rows * cols, -1))  # T*N*2
    x = []
    for t in range(min_t, max_t):
        x.append(data[t:t + 12])
    x = np.stack(x, axis=0)
    return x  # batch*T*N*2


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device(args.device)

    supports = None
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj, aptinit=adjinit,
                  onlyEMC=args.onlyEMC, onlyNMC=args.onlyNMC, onlyADP=args.onlyADP, hops=args.hops,
                  residual_channels=args.nhid,
                  dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location={args.oldGPU:args.device}))
    model.eval()

    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    testX = generate_x(1740, 1812)  # B*T*N*2
    testX = scaler.transform(testX)
    testX = torch.tensor(testX).to(device)
    testX = testX.transpose(1, 3)
    with torch.no_grad():
        testX = nn.functional.pad(testX, (1, 0, 0, 0))
        preds = model(testX).squeeze()  # batch*3*N
    preds = scaler.inverse_transform(preds)  # batch*3*N
    values = preds[:, 0, :]  # 72*N
    values = values.cpu().numpy()
    values = np.reshape(values, (-1, 32, 32))  # T*32*32
    np.savez_compressed(
        os.path.join(args.output_dir, args.plotName + ".npz"),
        preds=values,
    )

    # 第一个地点 R1[23,23] 邻近节点 [23,22] [23,24] [23,21] [23,25]


if __name__ == "__main__":
    main()
