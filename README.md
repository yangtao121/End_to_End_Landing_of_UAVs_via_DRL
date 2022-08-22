# End-to-End Landing of UAVs via Deep Reinforcement Learning

## Installation

### Step 1

```bash
mkdir e2e
cd e2e
git clone https://gitee.com/yangtaohome/RL
cd RL
pip install -e .
```

### Step 2

```bash
conda create -n RL python=3.8
conda activate RL
conda install tensorflow-gpu
conda install pybullet
```

### Step 3

```
cd ..
git clone https://github.com/yangtao121/End_to_End_Landing_of_UAVs_via_DRL
```

## Training

```bash
cd
cd e2e/End_to_End_Landing_of_UAVs_via_DRL/end2end
mpirun -n 4(depend your computer) python MPI_train2.py
```

