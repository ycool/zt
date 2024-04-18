SPLIT=mini
CHECKPOINT="/home/hujiangtao/github/navsim_workspace/exp/training_ego_mlp_agent/2024.04.17.15.46.33/lightning_logs/version_0/checkpoints/test_model.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  agent=ego_status_mlp_agent \
  agent.checkpoint_path=$CHECKPOINT \
  experiment_name=ego_mlp_agent \
  split=$SPLIT
