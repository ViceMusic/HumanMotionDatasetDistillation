# About this baseline

关于这个baseline，我们目前使用expmap格式的数据

将其转化为:

    Human3.6M 的 expmap 姿态转成 3D 关节点坐标

做了一个如下的转化形式

    99 维 expmap
    -> 32 个关节的 3D 坐标
    -> 取其中 22 个关节
    -> 66 维 xyz

这个模型的输入和输出为

    输入 shape:  [B, input_len, 66]
    输出 shape: [B, input_len, 66]
    最终取前 target_len 帧作为预测结果

性能的有关计算方式为：

对于预测值和真值：

    pred: [B, T, 22, 3]
    gt:   [B, T, 22, 3]

先算每个关节的 3D 距离：

    sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2 + (z_pred - z_gt)^2)

然后对 batch、时间、关节取平均。

代码中有一个设置，开启：

    config.use_relative_loss = True

还会额外加一个速度损失：

    pred[t] - pred[t-1]  vs  gt[t] - gt[t-1]

也就是不仅要求姿态位置对，还要求动作变化趋势也接近。最终大致是：

    loss = position_loss + velocity_loss

评估指标是人体动作预测里常用的 MPJPE：

    Mean Per Joint Position Error
    平均每关节位置误差，单位通常是毫米。

计算方式是：

    每个预测帧：
        对每个关节算 3D 欧氏距离
        对所有关节取平均
        对所有样本取平均

所以如果某一帧 MPJPE 是 45.2，意思是：

在这个未来时间点上，预测的人体关节平均偏离真实位置 45.2 毫米
原测试代码返回的是这些未来帧的误差：