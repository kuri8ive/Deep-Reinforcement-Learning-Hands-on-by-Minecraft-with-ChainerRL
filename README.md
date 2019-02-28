# Deep-Reinforcement-Learning-Hands-on-by-Minecraft-with-ChainerRL


## Minecraft

Do you know Minecraft? 

[PR movie](https://youtu.be/MmB9b5njVbA)

[Official Wiki](https://minecraft.gamepedia.com/Minecraft_Wiki) says as belows:

> [![Minecraft](https://gamepedia.cursecdn.com/minecraft_gamepedia/4/4d/Mclogo.svg?version=efb6f83c8e7d7bd4cdf0b73a2826c393)](https://minecraft.gamepedia.com/Minecraft) is a [sandbox](https://en.wikipedia.org/wiki/Open_world) construction game created by [Mojang AB](https://minecraft.gamepedia.com/Mojang_AB) founder [Markus "Notch" Persson](https://minecraft.gamepedia.com/Markus_Persson), inspired by *Infiniminer*, *Dwarf Fortress*, *Dungeon Keeper*, and Notch's past games *Legend of the Chambered* and *RubyDung*. Gameplay involves [players](https://minecraft.gamepedia.com/Player) interacting with the game world by placing and breaking various types of [blocks](https://minecraft.gamepedia.com/Block) in a [three-dimensional environment](https://minecraft.gamepedia.com/Overworld). In this environment, players can build creative structures, creations, and artwork on [multiplayer](https://minecraft.gamepedia.com/Multiplayer) servers and singleplayer worlds across multiple [game modes](https://minecraft.gamepedia.com/Gameplay).

This game has some characteristics related to today's reinforcement learning.

- can freely make buildings and other things
- can play various games
- can multiplay

These characteristics made Minecraft good material for studying reinforcement learning.

## Marlo Project

In this hands-on, you do deep reinforcement learning with Minecraft as a simulator environment. 
This time you use a Minecraft environment called [marLo](https://github.com/crowdAI/marLo) which is used in [MARLO](https://www.crowdai.org/challenges/marlo-2018), a contest utilizing Minecraft for a deep reinforcement learning. 
You can easily use a reinforcement learning framework such as ChainerRL because marLo is compatible with [OpenAI's Gym](https://github.com/openai/gym)(although it is not complete ... for example wrapper used for saving movies cannot be used).

MarLo has some environments as follows. For example, you can make an AI walking on a single road on lava with deep reinforcement learning. Now you learn with `MarLo-FindTheGoal-v0` environment, and then you try assignments.

## Hands-on content

This hands-on is based on [marlo-handson](https://github.com/keisuke-umezawa/marlo-handson).

## Requirements

As of December 14, 2018, the following is necessary.

Python 3.5+ environment with

- Chainer v5.0.0
- CuPy v5.0.0
- ChainerRL v0.4.0
- marlo v0.0.1.dev23

## Environment construction by Azure

In order to follow the following procedure, Azure subscription is required.

### 1. Installing Azure CLI

Please choose from the following according to your environment.

1. on Windows
   [Insatll Azure CLI on Windows](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?view=azure-cli-latest)

2. by Homebrew(macOS)

   ```bash
   $ brew update && brew install azure-cli
   ```

3. by Python

   ```bash
   $ pip install azure-cli
   ```

### 2. Log in to Azure

```bash
$ az login
```

### 3. Select subscription

With the following command, you can list the subscription you have.

```bash
$ az account list --all
```

Let's set the subscription you want to select for your account. Of course, replace [A SUBSCRIPTION ID] with your subscription ID.

```bash
$ az account set --subscription [A SUBSCRIPTION ID]
```

### 4. Start up the GPU VM

You create a data science VM. `--generate-ssh-keys` automatically creates the key to connect to the VM and saves it as secret key `id_rsa` and public key `id_rsa.pub` in `~ /.ssh/ `.

```bash
$ AZ_USER=[Any username you like e.g. kumezawa]
$ AZ_LOCATION=[location where resource-group was created e.g. eastus]
$ AZ_RESOURCE_GROUP=[resource-groupを作ったlocation e.g. marmo]
$ az vm create \
--location ${AZ_LOCATION} \
--resource-group ${AZ_RESOURCE_GROUP} \
--name ${AZ_USER}-vm \
--admin-username ${AZ_USER} \
--public-ip-address-dns-name ${AZ_USER} \
--image microsoft-ads:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest \
--size Standard_NC6 \
--generate-ssh-keys
```

If it works, you should see a message like below.

```bash
{
  "fqdns": "[YOUR USERNAME].eastus.cloudapp.azure.com",
  "id": "/subscriptions/[YOUR SUBSCRIPTION ID]/resourceGroups/marLo-handson/providers/Microsoft.Compute/virtualMachines/vm",
  "location": "eastus",
  "macAddress": "AA-BB-CC-DD-EE-FF",
  "powerState": "VM running",
  "privateIpAddress": "10.0.0.4",
  "publicIpAddress": "123.456.78.910",
  "resourceGroup": "marLo-handson",
  "zones": ""
}
```

Remember `publicIpAddress` for the next step.

#### Note

If the secret key `id_rsa` and the public key `id_rsa.pub` are in `~ /.ssh/`, the above command will cause an error.
In that case, you can create your own key and designate it to use.

```bash
$ az vm create \
--location ${AZ_LOCATION} \
--resource-group ${AZ_RESOURCE_GROUP} \
--name ${AZ_USER}-vm \
--admin-username ${AZ_USERER} \
--public-ip-address-dns-name ${AZ_USER} \
--image microsoft-ads:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest \
--size Standard_NC6 \
--ssh-key-value [Public key path e.g. ~/.ssh/id_rsa.pub]
```

#### Note

If you want to start up on a CPU instance instead of a GPU, try `--size Standard_D2s_v3` for example. If you want to check the size of other available VMs, you can look it up like this.

```bash
az vm list-sizes --location eastus --output table
```

### 5. Open the port required for access

```bash
$ az vm open-port --resource-group ${AZ_RESOURCE_GROUP} --name ${AZ_USER}-VM --port 8000 --priority 1010 \
&& az vm open-port --resource-group ${AZ_RESOURCE_GROUP} --name ${AZ_USER}-VM --port 8001 --priority 1020 \
&& az vm open-port --resource-group ${AZ_RESOURCE_GROUP} --name ${AZ_USER}-VM --port 6080 --priority 1030
```

### 6. ssh connection to VM

```bash
$ AZ_IP=[your VM IP e.g. "40.121.36.99"]
$ ssh ${AZ_USER}@${AZ_IP} -i ~/.ssh/id_rsa
```

### 7. Create a Conda environment for MarLo

Execute the following command in VM environment.

```bash
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
&& sudo apt-get update \
&& sudo apt-get install -y libopenal-dev
```

```bash
$ conda config --set always_yes yes \
&& conda create python=3.6 --name marlo \
&& conda config --add channels conda-forge \
&& conda activate marlo \
&& conda install -c crowdai malmo matplotlib ipython numpy scipy opencv \
&& pip install git+https://github.com/crowdAI/marLo.git \
&& pip install chainer==5.1.0 cupy-cuda92==5.1.0 chainerrl==0.5.0
```

### 8. Starting Minecraft client with Docker

```bash
$ sudo docker pull ikeyasu/marlo:latest
$ VNC_PW=[Any password you like]
$ sudo docker run -it --net host --rm --name robo6082 -d -p 6080:6080 -p 10000:10000 -e VNC_PW=${VNC_PW} ikeyasu/marlo:latest
```

Please go to `http://your vm IP:6080` and enter your own password `$ {VNC_PW}`.

You should be able to connect to the remote environment via a VNC connection and check that Minecraft will start in that!



## Let's start hands-on

### 0. Check if the Conda environment is activated

```bash
$ conda info -e
# conda environments:
#
base                     /anaconda
marlo                 *  /anaconda/envs/marlo
py35                     /anaconda/envs/py35
py36                     /anaconda/envs/py36
```

If the `marlo` created earlier is not activated, please execute the command below.

```bash
$ conda activate marlo
```

### 1. Clone Hands-on repository

```bash
$ git clone https://github.com/keisuke-umezawa/marlo-handson.git
$ cd marlo-handson
```

### 2. Run malro test script

```bash
$ python test_malmo.py
```

After waiting for a while, you should see a screen like this.

The above command executes the following python script.

[test_malmo.py](https://github.com/keisuke-umezawa/marlo-handson/blob/master/test_malmo.py)

```python
import marlo


def make_env(env_seed=0):
    join_tokens = marlo.make(
        "MarLo-FindTheGoal-v0",
        params=dict(
            allowContinuousMovement=["move", "turn"],
            videoResolution=[336, 336],
            kill_clients_after_num_rounds=500
        ))
    env = marlo.init(join_tokens[0])

    obs = env.reset()
    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    env.seed(int(env_seed))
    return env


env = make_env()
obs = env.reset()


for i in range(10):
    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    print(r, done, info)
```

- changing "MarLo-FindTheGoal-v0" can change the environment.
- obs is numpy format image data.
- r is the reward for the previous action.
- done is a boolean value whether the game is over.
- info contains other information.

#### Assignment 1: Check or change the above.

### 3. Execution of training script of DQN by ChainerRL

Execute the following command first. You can start training on reinforcement learning model DQN from scratch with ChainerRL.

```bash
$ python train_DQN.py
```

Stop the script by pressing CTRL + C whenever you like. Then you should have the following directories. The model trained is stored in xxxx_except.

#### Note

If you want to run by cpu only, add the following option.

```bash
$ python train_DQN.py --gpu -1
```

### 3. Check the operation of the saved model

You can load the trained model and check its operation with the following command.

```bash
$ python train_DQN.py --load results/3765_except --demo
```

What's the result look like? The model has not trained much, so it should not work properly.

### 4. Start training from the saved model

You can resume training from a previously saved model by the following command:

```bash
$ python train_DQN.py --load results/3765_except
```

You also can use a prepared model which is already trained to some extent. However, it is a model created by simply running `train_DQN.py`, so it may not work if you change the code or the environment.

```bash
$ wget https://github.com/keisuke-umezawa/marlo-handson/releases/download/v0.2/157850_except.tar.gz
$ tar -xvzf 157850_except.tar.gz
$ python train_DQN.py --load 157850_except
```

#### Assignment 2: Read the source code of train_DQN.py

[train_DQN.py](https://github.com/keisuke-umezawa/marlo-handson/blob/master/train_DQN.py)

```python
# ...ellipsis...

def main():
    parser = argparse.ArgumentParser()

    # ...ellipsis...

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    experiments.set_log_base_dir(args.out_dir)
    print('Output files are saved in {}'.format(args.out_dir))

    env = make_env(env_seed=args.seed)

    n_actions = env.action_space.n

    q_func = links.Sequence(
        links.NatureDQNHead(n_input_channels=3),
        L.Linear(512, n_actions),
        DiscreteActionValue
    )

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=args.lr, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        x = x.transpose(2, 0, 1)
        return np.asarray(x, dtype=np.float32) / 255

    agent = agents.DQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator='sum',
        phi=phi
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.out_dir,
            save_best_so_far_agent=False,
            max_episode_len=args.max_episode_len,
            eval_env=env,
        )


if __name__ == '__main__':
    main()
```

References

- [NatureDQNHead](https://github.com/chainer/chainerrl/blob/v0.5.0/chainerrl/links/dqn_head.py#L13-L36)
- [DiscreteActionValue](https://github.com/chainer/chainerrl/blob/v0.5.0/chainerrl/action_value.py#L53-L102)
- [ReplayBuffer](https://github.com/chainer/chainerrl/blob/v0.5.0/chainerrl/replay_buffer.py#L131-L163)
- [LinearDecayEpsilonGreedy](https://github.com/chainer/chainerrl/blob/v0.5.0/chainerrl/explorers/epsilon_greedy.py#L51-L90)
- [DQN](https://github.com/chainer/chainerrl/blob/v0.5.0/chainerrl/agents/dqn.py#L84)

#### Assignment 3: Improve Performance

There are some ways to improve the performance of reinforcement learning. For example, 

- change model
- change ReplayBuffer
- changing parameters

And there is a [paper](https://arxiv.org/pdf/1710.02298.pdf) which evaluated performance improvement by trying various reinforcement learning methods. According to this, you may improve performance by the following:

- use PrioritizedReplayBuffer
- use DDQN
