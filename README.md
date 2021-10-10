# Mutil-branch-SENet
# 使用数据
来源：中科院皮研所(HE, 119, WSI, MF)
标注：Congcong Xu
修正：Junjie Wang
Train: 464
Val:112
Test:77(WSI)
# 评价指标
评估主要分为两个方面：
1：消融分支实验中，分别对两种细胞类别进行Accuracy，F1，IoU，Perfermance，Precision，Recall的评估(Paper中展示为淋巴样细胞评估结果(因为医生更关心淋巴样细胞))；
2：对比实验中，再加上识别率指标，因为网络最终目的是分类两种细胞，上皮细胞的分类准确率对于任务本身而言同样重要。
# 评价参考
Accuracy，F1，IoU，Perfermance，Precision，Recall:Improving Nuclei/Gland Instance Segmentation in Histopathology Images by Full Resolution Neural Network.
code:https://github.com/huiqu18/FullNet-varCE
识别率：Concurrency and Comput- ation: Practice and Experience
# 评估结果
消融实验：

评价指标	主任务分支	主任务+实例分支	主任务＋边界分支	原模型
Accuracy	0.940	0.930	0.940	0.943
F1	0.700	0.708	0.727	0.738
IoU	0.570	0.548	0.571	0.580
Performance[22]	1.700	1.685	1.700	1.735
Precision	0.720	0.701	0.727	0.711
Recall	0.730	0.716	0.727	0.760

对比实验：
网络	识别率	IoU	F1	Precision	Recall	Accuracy
Unet[24]	0.863	0.670	0.700	0.720	0.730	0.833
Unet-3	0.878	0.648	0.708	0.701	0.716	0.837
Hover-Net	0.949	0.671	0.727	0.727	0.727	0.933
Ours	0.943	0.680	0.738	0.711	0.760	0.943

