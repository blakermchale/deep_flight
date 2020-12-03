# Deep Flight

Final Project for CS4100 Artificial Intelligence  
Professor Gold

<img src="docs/DeepFlight.png" height="700">

## Install

Setup AirSim with Unreal Engine 4.24 from their [tutorial](https://microsoft.github.io/AirSim/build_windows/)

``` bash
sudo apt install git-lfs
git lfs install
```

Setting up python:  
``` powershell
pip3 install msgpack-rpc-python airsim gym tensorflow Pillow
pip3 install -e .
```

Add user AirSimPath environment variable. Look up environment variables on windows [(Link)](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/)  
Run update_from_git.bat to setup AirSim plugin in environment. Should be done in powershell.  
``` powershell
cd Environments\Blocks
.\update_from_git.bat $Env:AirSimPath
```

## Running

Copy `settings.json` to Documents\AirSim.
``` powershell
cp .\settings.json $HOME\Documents\AirSim\
```

Open [Environments/Blocks/Blocks.sln](Environments/Blocks/Blocks.sln) in visual studio. Press `F5`.

```powershell
python dqn_agent.py
```

## Errors

**ERROR:** 'C:\Program Files\Epic Games\4.24\Engine\Intermediate\Build\Unused\UE4.exe does not exist'
    when running debug editor in VS  
**[SOLUTION](https://answers.unrealengine.com/questions/218266/unable-to-start-program-ue4exe-error.html)**

**ERROR:** pip install tensorflow cannot find file called client_load_reporting_filter.h  
**[SOLUTION](https://docs.python.org/3/using/windows.html#removing-the-max-path-limitation)**

## Resources

[Our Google Drive](https://drive.google.com/drive/u/1/folders/1pUoStc6qpU9d41R7aaGgp70TfvcZjjC6)

[DQN with Keras](https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c)  
[Official AirSim DQN](https://github.com/microsoft/AirSim/blob/d59ceb7f63878f5e087ea802d603ba0fd282ff56/PythonClient/multirotor/DQNdrone.py)  
[AirGym](https://github.com/Kjell-K/AirGym)  
[Another Paper Ex](https://www.mdpi.com/2076-3417/9/24/5571/htm)
