# VOC 1-way 1-shot

python train.py with gpu_id=0 mode='train' dataset='VOC' label_sets=3  task.n_ways=1 task.n_shots=1 
python test.py with gpu_id=0 mode='test' snapshot='./runs/SMCP_VOC_sets_3_1way_1shot_[train]/1/snapshots/39000.pth'
