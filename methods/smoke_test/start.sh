# 启动环境
#!/usr/bin/env bash
set -e

eval "$(mamba shell hook --shell bash)"
mamba activate motiondd

#nohup python run_HDT_human36m_temp.py > run_HDT_human36m_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#nohup python train.py > distill_test_hdt_h001_g01_bs64_iter8000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#nohup python train_regular_HDT_distill_simple_human36m.py > distill_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# nohup python run_sequence_hdt_distill_and_train_noharm.py > run_sequence_hdt_distill_and_train_noharm.log 2>&1 &
# nohup python run_sequence_hdt_distill_and_train.py > run_sequence_hdt_distill_and_train.log 2>&1 &
# nohup python random_baseline.py > random_baseline.log 2>&1 &  # 随机

#nohup python run_sequence_hdt_distill_and_train_realinit_fixedbackbone.py > run_sequence_hdt_distill_and_train_realinit_fixedbackbone.log 2>&1 & # 真实init 和 backbone固定

# 知识蒸馏
# nohup python run_privileged_future_distill_and_train.py > run_privileged_future_distill_and_train.log 2>&1 &
# nohup python run_privileged_future_stgraph_distill_and_train.py > run_privileged_future_stgraph_distill_and_train.log 2>&1 &  # 初步修改
nohup python run_privileged_future_stgraph_weighted_distill_and_train.py > run_privileged_future_stgraph_weighted_distill_and_train.log 2>&1 &