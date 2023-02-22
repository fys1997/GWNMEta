Final Code for GWN

python train.py --device cpu --gcn_bool --aptonly --addaptadj --randomadj --batch_size 1 --print_every 1
python test.py --device cuda:3 --gcn_bool --aptonly --addaptadj --randomadj --batch_size 64 --checkpoint modelSave/master/BJ-Flow.pth  --num_nodes 1024 --plotheatmap False --data data/BJ-Flow