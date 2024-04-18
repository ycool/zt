set -x
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  experiment_name=training_ego_mlp_agent

#  +agent=ego_status_mlp_agent \
  # --info defaults