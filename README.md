# 没什么用的启动指令
mkdir -p ~/.ssh
chmod 700 ~/.ssh

curl -L https://github.com/xarnudvilas.keys >> ~/.ssh/authorized_keys

sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 数据位置以及格式读取方式

npz格式数据的绝对地址

 /home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz

npz格式数据的封装说明

    subjects	    每段动作属于哪个, 例如 S1, S5, S6,其中包括字段为:S1,11,5,6,7,8,9
    actions	        动作类别名称, 例如 directions, discussion, eating
    trials	        动作编号, 例如 directions_1.txt 对应 1
    lengths	        每段动作的帧数， 例如某段动作有 1383 帧
    raw_paths	    原始 .txt 文件路径, 用于追踪原始数据来源
    motions	        每段动作的实际数据,每个元素是一个 [T, 99] 的 numpy array
    feature_type	特征类型, 当前为 "expmap"
    feature_dim	    每帧特征维度, 当前为 99

    其中最重要的是：

        motion = data["motions"][i]

    它会得到第 i 段连续动作序列，形状为：

        [T, 99]

## ETT

数据位置为：/home/user/workspace/HumanMotionDatasetDistillation/datasets/raw/ETT, 包括如下文件:
* ETTh1.csv
* ETTh2.csv
* ETTm1.csv
* ETTm2.csv

数据格式以ETTh1.csv的前五行为例子

    date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
    2016-07-01 00:00:00,5.827000141143799,2.009000062942505,1.5989999771118164,0.4620000123977661,4.203000068664552,1.3400000333786009,30.5310001373291
    2016-07-01 01:00:00,5.692999839782715,2.075999975204468,1.4919999837875366,0.4259999990463257,4.142000198364259,1.371000051498413,27.78700065612793
    2016-07-01 02:00:00,5.1570000648498535,1.741000056266785,1.2790000438690186,0.35499998927116394,3.776999950408936,1.218000054359436,27.78700065612793
    2016-07-01 03:00:00,5.0900001525878915,1.9420000314712524,1.2790000438690186,0.3910000026226044,3.806999921798706,1.2790000438690186,25.0440006256103


