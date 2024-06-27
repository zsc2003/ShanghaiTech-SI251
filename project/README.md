# ShanghaiTech-SI251-Final-Project
ShanghaiTech SI251 Convex Optimization final project, Spring 2024.




## Train pointnet
```bash
cd pointnet
CUDA_VISIBLE_DEVICES=0 python main.py --outlier_fraction 0.6 --robust_type 'H' --alpha 1.0
```

robust type could be modified:
- Q: quadratic
- PH: pseudo-Huber
- H: Huber
- W: Welsch
- TQ: truncated quadratic
- None: default, max-pooling