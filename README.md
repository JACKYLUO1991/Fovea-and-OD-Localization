自动导出依赖项：
    pip freeze > environment.txt
    conda env export > environment.yaml

增加空间：
    ulimit -SHn 51200

改进的方案：

（1）利用平均池化代替最大池化；

（2）soft-argmax代替argmax，力求可微分：https://kornia.readthedocs.io/en/v0.1.3/_modules/kornia/geometry/dsnt.html#spatial_softmax_2d；

（3）