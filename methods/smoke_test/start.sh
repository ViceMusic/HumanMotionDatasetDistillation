#nohup python single_fixed_backbone.py > distill_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python train.py > distill_test_full_111_bs64_iter8000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#nohup python train_regular_HDT_distill_simple_human36m.py > distill_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &