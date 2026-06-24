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

# nohup python run_sequence_distill_and_train_realinit_fixedbackbone.py > run_sequence_distill_and_train_realinit_fixedbackbone.log 2>&1 & # 真实init 和 backbone固定

# 知识蒸馏
# nohup python run_privileged_future_distill_and_train.py > run_privileged_future_distill_and_train.log 2>&1 &
# nohup python run_privileged_future_stgraph_distill_and_train.py > run_privileged_future_stgraph_distill_and_train.log 2>&1 &  # 初步修改
#nohup python run_privileged_future_stgraph_weighted_distill_and_train.py > run_privileged_future_stgraph_weighted_distill_and_train.log 2>&1 &

#python run_dlinear_full_data_train_eval.py > run_dlinear_full_data_train_eval.log 2>&1 & # 完整Dlinear数据集

#nohup python random_baseline.py > random_baseline.log  2>&1 &

# nohup python run_sequence_dlinear_distill_random_compare.py > run_sequence_dlinear_distill_random_compare.log  2>&1 &

#nohup python run_sequence_simlpe_adaptive_actions.py > run_sequence_simlpe_adaptive_actions.log 2>&1 &

#nohup python run_sequence_dlinear_puredistill_random_compare.py > run_sequence_dlinear_puredistill_random_compare.log 2>&1 &

#nohup python run_sequence_maam_gm_distill_and_train.py > run_sequence_maam_gm_distill_and_train.log 2>&1 &

# 完全纯粹的蒸馏（没有任何速度之类的限制）
# nohup python run_sequence_simple_puredistill_and_train.py > run_sequence_simple_puredistill_and_train.log 2>&1 &


# 平行跑一波对比

# nohup python run_segment_selection_gm_parallel.py > run_segment_selection_gm_parallel.logs 2>&1 &

# 平行跑一波mean+std
# nohup python run_gm_seed_sweep_parallel.py > run_gm_seed_sweep_parallel.log 2>&1 &
# 平行跑一下上面这条指令的baseline
# nohup python run_gm_seed_sweep_baseline_parallel.py > run_gm_seed_sweep_baseline_parallel.log 2>&1 
# 对Dlinear做类似的测试
# nohup python run_dlinear_seed_sweep_full_random_gm.py > run_dlinear_seed_sweep_full_random_gm.log 2>&1 &
# 修正一下做做尝试
# nohup python run_dlinear_seed_sweep_full_random_gm_protocol_fixed.py > run_dlinear_seed_sweep_full_random_gm_protocol_fixed.log 2>&1 &

# 一个残差，不确定有用
# nohup python run_map_gm_middle_anchor.py > run_map_gm_middle_anchor.log 2>&1 &

# 在纯净蒸馏的基础上加强动作约束

#nohup python run_sequence_simple_distill_and_train_300weight.py > run_sequence_simple_distill_and_train_300weight.log 2>&1 &


# 检查一下Dlinear的结果
# run_dlinear_msweep_plus.py
# nohup python run_dlinear_msweep_plus.py > run_dlinear_msweep_plus.log 2>&1 &



nohup python run_dlinear_msweep_len200.py  > logs/nohup_dlinear_msweep_len200.out 2>&1 &
#nohup python run_dlinear_msweep_len70.py  > logs/nohup_dlinear_msweep_len70.out 2>&1 &
# 下面这个执行完成了
#nohup python run_dlinear_msweep_len100_batchmix.py  > logs/nohup_dlinear_msweep_len100_batchmix.out 2>&1 &

# 先不跑下面这个
# nohup python methods/smoke_test/run_dlinear_msweep_in20_out10_len40_batchmix.py  logs/nohup_dlinear_msweep_in20_out10_len40_batchmix.out 2>&1 &