# FaultTolerance_AACFT

## Introduction
This repository contains the code for the platform mentioned in our research paper [Toward Fault Tolerance in Multi-Agent Reinforcement Learning](https://ieeexplore.ieee.org/document/11096942) along with the proposed AACFT algorithm.


## Usages
To use the platform with your own scripts, follow these steps:
- Place the algorithm script in the `ftcode/algorithms` folder, the fault script in the `ftcode/fault` folder, and the scenario script in the `mpe/scenarios` folder.
- Store the .yaml files in the `ftcode/configs` folder for configuration of algorithms and environments.
- Run the command for training:
```
python ftcode/main.py --env your_env_name --alg your_algorithm_name --fault your_fault_name
```
For example:
```
python ftcode/main.py --env fix --alg aacft --fault broken --flag 10 --device cuda:0 --seed 0 --debug
```

- Run the command for testing:
```
python ftcode/main.py --env your_env_name --alg your_algorithm_name --fault your_fault_name --old_model_name your_model_name --test
```
For example:
```
python ftcode/main.py --flag 10 --device cuda:0 --env fix --alg aacft --fault broken --old_model_name fix_aacft_seed_time_broken/100000 --test
```
- Run the command for testing and environment rendering:
```
python ftcode/main.py --env your_env_name --alg your_algorithm_name --fault your_fault_name --old_model_name your_model_name --test --display --n_rollout_threads 1
```

## Citation
```
@ARTICLE{11096942,
  author={Shi, Yuchen and Pei, Huaxin and Feng, Liang and Zhang, Yi and Yao, Danya},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={Toward Fault Tolerance in Multi-Agent Reinforcement Learning}, 
  year={2025},
  volume={22},
  number={},
  pages={19007-19024},
  doi={10.1109/TASE.2025.3592721}}
```