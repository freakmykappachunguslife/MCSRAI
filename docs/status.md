---
layout: default
title: Status
---

## Project Summary

Our goal has shifted from getting an agent to speedrun Minecraft, eventually seeing the end credits scene after defeating the ender dragon, to being able to enter the nether.

So far, given the materials, our agent has been able to build the nether portal frame, light it, and enter the nether.

However, we want this agent to be able to spawn in a world and be able to enter the nether itself. Given that we were able to achieve the building of the portal, our current objective is to get the agent to navigate and build the portal from a set seed.

We chose to focus on a set seed as we think it would be simpler to allow it to converge on a "best" solution, but we plan to attempt random seeds as well later in the project if possible.

## Approach
With MineRL and MineDojo, we are able to create an environment that our model can run on; the model can "play" Minecraft, making whatever inputs it chooses.

The state space includes the RGB information of each frame, the inventory, and the voxel space, which provides information about blocks in a 9x9 area around the agent. 

Attached below is an example of an agent on a random seed making random inputs (be warned there are flashing lights).
<video controls src="videos/randomagent.mp4" type="video/mp4" width="600" height="400">
    Your browser does not support the video tag.
</video>
You can see that the agent does manage to get wheat seeds, but that is simply random chance. The agent does a whole lot of nothing. The most glaring issue with the random agent is that it is, well, random; often, it opens the inventory, and proceeds to randomly press buttons and moves the mouse, which does nothing for approaching the end goal. 

So, we had to redefine the action space, heavily punishing the agent for inputting item drops, switching hands, and opening the inventory. 

We train our agent using Proximal Policy Optimization PPO from Stable Baselines3. PPO collects rollout data from the environment and updates the policy using a clipped objective to ensure stable learning. 

We trained for about 500,000 interaction steps using mostly default PPO hyperparameters, including learning rate = 0.0003, gamma = 0.99, batch size = 256, and clip range = 0.2. Rewards are milestone based, encouraging obsidian placement, frame completion, portal lighting, and nether entry.

Our agent learned how to build, light, and enter a nether portal. Now, we are training it to successfully achieve this on a set seed.

Below is the video of our best agent completing the nether portal, given the obsidian and flint and steel.

<video controls src="videos/buildportal.mp4" type="video/mp4" width="600" height="400">
    Your browser does not support the video tag.
</video>

The agent is pretty efficient at creating the portal!


## Evaluation
For the building portal agent, we evaluated on three "milestones", completion of a portal frame, lighting of the portal, and entering the portal. The lighting and entering are boolean values which we can easily identify, as entering the nether changes the "dimension" state, and the lighting creates "portal" blocks. Evaluating the portal frame is difficult, as Minecraft does allow portals to be any rectangular size, with a minimum frame size of 4x5, but we only want the 4x5 size.

The agent is rewarded for obtaining obsidian and placing obsidian, as well as lighting and entering the portal. We want the total reward to converge at a certain value whilst achieving the milestones so the agent is taking the most efficient path towards entering the nether (not placing extra obsidian or making extra portals). Furthermore, we are ranking our agents on time, the faster the time, the better the agent.

This allowed us to create an agent that very efficiently creates a nether portal (as seen in the approach section).

However, there is an issue. Because we trained exclusively on flat terrain, this “best” agent cannot guarantee the creation of the nether portal on non-flat terrain. This will have to be addressed in our non-flat world runs.


<video controls src="videos/buildportalbad.mp4" type="video/mp4" width="600" height="400">
    Your browser does not support the video tag.
</video>

For the future, we plan on using behavioural training/imitation learning as MineDojo contains video data sets.


## Remaining Goals and Challenges

Currently, we are continuing to improve our agent's capability of running the set seed.


We still plan to train our agent to enter the nether on random seeds, and if there's time, complete speedruns. However, this will still take a lot of time and we are unsure if we will be able to achieve this.


## Resources Used

Our main repository can be found at this link:
[Main Repo](https://github.com/freakmykappachunguslife/MCSRAI)


The main tools we used was MineRL and MineDojo, both created for Minecraft AI:
[MineDojo](https://minedojo.org)


[MineRL](https://minerl.readthedocs.io/en/latest/index.html)




## Video Summary





