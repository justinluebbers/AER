# AER
Bachelor's Thesis: Augmented Experience Replay - Imaginative Data Augmentation for Deep Reinforcement Learning

This is the complementary code repository to our thesis: Augmented Experience Replay - Imaginative Data Augmentation for Deep Reinforcement Learning. 

---Folder structure---
- AER
    - Code: All the code files used for our implementation
    - data: Expert data files
    - logs: Logs from experiments
    - model_training: loss records from training the environment models
    - models: Environment models as well as control models
    - plots: Various plots from running experiments
    - scores: Score records collected during experiments
- baselines: baselines package containing the enhanced deepq implementation


---Code files---
- aer: Implementation of central imagination component of AER
- approximator: Implementation of environment models
- data_manager: Utility class that manages data access for model training
- DummyAcrobotEnv: Playground copy of Acrobot environment
- DummyCartPoleEnv: Playground copy of CartPole environment
- DummyMountainCarEnv: Playground copy of MountainCar environment
- env_training_parameters: Contains experiment settings such as the used environment model
- get_plots: Utility functions for generating score and learning curve plots used in the thesis
- run_aer: Convenient function to run AER experiments, including benchmark DQN
- train_approximator: Convenient function to run model training
- util: Utility function to ease terminal execution


---Usage---
AER experiments:
To recreate the AER experiments from our thesis, run the following command: 

python -W ignore ./AER/Code/run_aer.py --env=MountainCar --alg=deepq --iterations=5 --rollouts=[16x1,32x1,48x1,2x8,4x8,6x8,1x16,1x32,1x48,2x16,3x16,2x32,4x4,8x4,12x4] --real_batch_sizes=[32,48,64,80] 

The -W ignore option disables TensorFlow warnings. For specific experiments simply change the correspondent argument


Model training:
To train environment models run

python -W ignore ./AER/Code/train_approximator.py --n_steps=20000 --hwidths=[32,64] --hdepths=[4,8] --model_name=EM_MountainCar --data_filename=MountainCar_transitions_1000000.p --loss_function=nrmse_range --suc_ratio=0.0

This will train models for all combinations of widths and dephts specified in the arguments


Expert data collection:
To gather expert data run

python -W ignore ./baselines/baselines/run.py --env=MountainCar --alg=deepq --num_timesteps=1000000 --save_buffer=./AER/data/MountainCar_transitions_1000000.p


Generate plot:
To generate plots for the experiments run

python -W ignore ./AER/Code/get_plots.py --target=curves --env=MountainCar

target can either be curves or scores


