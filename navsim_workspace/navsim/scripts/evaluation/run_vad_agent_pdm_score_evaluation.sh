SPLIT=mini
CHECKPOINT="/home/hujiangtao/github/zt/navsim_workspace/exp/vad_agent/model/VAD_tiny.pth"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  agent=vad_agent \
  agent.checkpoint_path=$CHECKPOINT \
  experiment_name=vad_agent \
  split=$SPLIT
