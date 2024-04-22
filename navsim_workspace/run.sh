# set -x

export HYDRA_FULL_ERROR=1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/" && pwd -P)"

export PYTHONPATH=$PYTHONPATH:${DIR}/navsim:${DIR}/nuplan-devkit

if [ $# -lt 1 ]; then
  echo "run.sh [cmd]"
  echo "cmd: notebook, cache, training, eval_cv, eval_human, eval_mlp"
  exit
fi

case $1 in
    "notebook")
        # visualization
        # select kernel to use navsim which is conda python kernel and contains conda packages
        jupyter notebook &
        ;;
    "cache")
        # preprocess & cache
        # output: exp/metric_cache
        ./navsim/scripts/evaluation/run_metric_caching.sh
        ;;
    "training")
        # training
        ./navsim/scripts/training/run_ego_mlp_agent_training.sh
        ;;
    "eval_cv")
        # const velocity agent evaluation
        ./navsim/scripts/evaluation/run_cv_pdm_score_evaluation.sh
        ;;

        # output
        # [hujiangtao@dream:  ~/github/navsim_workspace/exp/cv_agent/2024.04.16.17.46.45 ] conda:base
        # ConstantVelocityAgent_2024.04.16.17.51.26.csv
        # seq,token,valid,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,ego_progress,time_to_collision_within_bound,comfort,score
        # 3614,dd1e26943afe52bc,True,1.0,1.0,1.0,0.8906223480884186,1.0,1.0,0.9544259783701744
        # 3615,average,True,0.9258644536652836,0.9156293222683264,0.9991701244813278,0.7556808946641205,0.8816044260027662,1.0,0.8013567404824862
    "eval_human")
        # human agent
        ./navsim/scripts/evaluation/run_human_agent_pdm_score_evaluation.sh
        # output: /github/navsim_workspace/exp/human_agent/2024.04.16.17.56.36
        # HumanAgent_2024.04.16.18.01.19.csv
        # seq,token,valid,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,ego_progress,time_to_collision_within_bound,comfort,score
        # 3614,dd1e26943afe52bc,True,1.0,1.0,1.0,0.8350611325112367,1.0,1.0,0.931275471879682
        # 3615,average,True,0.9955739972337483,0.9817427385892116,0.9991701244813278,0.8716685864329,0.9786998616874135,0.9988934993084371,0.9259548376197573
        ;;

    "eval_mlp")
        # ego mlp agent
        ./navsim/scripts/evaluation/run_ego_mlp_agent_pdm_score_evaluation.sh
        ;;
    *)
        echo "run.sh [cmd]"
        echo "cmd: notebook, cache, training, eval_cv, eval_human, eval_mlp"
        ;;
esac

# add ~/.bashrc
# export NUPLAN_MAPS_ROOT="$HOME/github/navsim_workspace/dataset/maps"
# export NAVSIM_EXP_ROOT="$HOME/github/navsim_workspace/exp"
# export NAVSIM_DEVKIT_ROOT="$HOME/github/navsim_workspace/navsim"
# export OPENSCENE_DATA_ROOT="$HOME/github/navsim_workspace/dataset"

# mkdir ~/.pip
# cat <<EOF > ~/.pip/pip.conf
#  [global]
#  trusted-host =  mirrors.aliyun.com
#  index-url = http://mirrors.aliyun.com/pypi/simple
# EOF

# rsync -ahvPz maps zt@172.25.117.74:~/cvpr/zt/navsim_workspace/dataset

# install conda
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh

# conda env create --name navsim -f environment.yml
# conda activate navsim
# pip install -e .
# installed dir: /home/zt/miniconda3/envs/navsim/lib/python3.9/site-packages

# install jupyter notebook for conda env
# conda install -c conda-forge notebook
# conda activate navsim
# conda install ipykernel
# python -m ipykernel install --user --name=navsim



