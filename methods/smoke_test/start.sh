#nohup python run_HDT_human36m_temp.py > run_HDT_human36m_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python train.py > distill_test_hdt_h001_g01_bs64_iter8000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#nohup python train_regular_HDT_distill_simple_human36m.py > distill_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &