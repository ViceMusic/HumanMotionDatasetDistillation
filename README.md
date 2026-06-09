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


