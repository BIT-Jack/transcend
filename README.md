
# Introduction
Official codes of paper **_Bio-inspired Task-free Continual Learning for Autonomous Driving in Interactive Scenarios: A Lifelong Scene Understanding Paradigm_**, which has been submitted to _IEEE Transactions on Robotics_.

# Processed Data
The experiments in this work are based on [INTERACTION dataset](https://interaction-dataset.com/) (Version: INTERACTION-Dataset-DR-single-v1_2).
The processed data is available in this link for [download](https://drive.google.com/drive/folders/1roEeNQJFz777DbPEMf21R3j2BQdRKecp?usp=drive_link).

# Implementations
## Enviroment
1. System: The codes can be run in **Ubuntu 22.04 LTS**.
2. **Python = 3.9**
3. **Pytorch = 2.0.0**
4. Other required packages are provided in "**requirements.txt**":
```
 pip install -r requirements.txt
```
## Configurations
1. Before running the codes, please revise "**root_dir**" in "_./utils/args_loading.py_" to your local paths.
2. Hyper-parameters for the approach can be also revised in "_./utils/args_loading.py_".

# Running

## Key Parameters for Training and Testing
1. **--model**: the name of the approaches that you want to train. Available names can be found in the filefolder "_./cl_model/_". 
2. **--buffer_size**: the amount of memory samples for continual learning approaches. Please set the buffer size as "0" when using the vanilla.
3. **--dataset**: set as "seq-interaction" when continual training, and set as "joint-interaction" when joint training.
4. **--train_task_num**: the total number of tasks in lifelong learning tasks.
5. **--debug_mode**: _True_ or _1_ when you are debugging, only a few batches of samples will be used in each task for a convenient check. _False_ or _0_ in the formal training.  
6. **--num_tasks**: the number of tasks to be tested. 

The file "_./bash_training_and_test.sh_" is an running example, showing the implementation of abovementioned parameters.
The default values of these key parameters can be revised in "_./utils/args_loading.py_".

## Simple usage of the bash file
After adding the executable permissions to the provided bash file (_bash_training_and_test.sh_) and entering the required running environment, you can directly run the training and testing with the following command:
```
./bash_training_and_test.sh
```

## Experimental results and recording
After training, the model weights will be saved in "_./results/weights/_".

After testing, the tested evaluation metrics will be saved in "_./results/logs/_". 

## Other tips
1. If you want to record the memory samples at each step of memory buffer update, you can turn **--replayed_rc** as _True_ or _1_ (the default value is _False_).
   The recording will be saved in "_./logging/replayed_memory/_" as .txt files, which records the task label of selected memory samples at each step.
2. **--store_traj** is the another optional parameter to record predicted heatmap and endpoints for visualization. The default value is _False_.
3. When training the model 'gem', the code may pause due to the failed QP solution. For a efficient restart, you can use **--restart_training** and **--restart_pre_task_num** to load trained weights and continue the lifelong learning.
   Specifically, turn **--restart_training** into True, and set the **--restart_pre_task_num** as the id of the lateset finished task.

*Note that please DELETE additional parameters (bool type) **--replayed_rc**, **--store_traj**, and **--restart_training** in .sh files, if you DO NOT use them.








