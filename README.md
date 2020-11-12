# deep_flight
Final Project for CS4100 Artificial Intelligence with Professor Gold

## Install

``` bash
sudo apt install git-lfs
git lfs install
```

Setting up python:

``` cmd
pip3 install msgpack-rpc-python airsim gym
```

Add user AirSimPath environment variable. Look up environment variables on windows [(Link)](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/)

Run update_from_git.bat to setup AirSim plugin in environment. Should be done in powershell.

``` cmd
.\update_from_git.bat $Env:AirSimPath
```

## Running

Copy `settings.json` to Documents\AirSim.

Open [Environments/Blocks/Blocks.sln] in visual studio. Press `F5`.

## Errors

**ERROR:** 'C:\Program Files\Epic Games\4.24\Engine\Intermediate\Build\Unused\UE4.exe does not exist'
    when running debug editor in VS

**[SOLUTION](https://answers.unrealengine.com/questions/218266/unable-to-start-program-ue4exe-error.html)**
