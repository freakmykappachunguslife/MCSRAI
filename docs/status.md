---
layout: default
title: Status
---

## Project Summary

Our goal has shifted from getting an agent to speedrun Minecraft, eventually seeing the end credits scene after defeating the ender dragon, to being able to enter the nether.

So far, given the materials, our agent has been able to build the nether portal frame, light it, and enter the nether. 

However, we want this agent to be able to spawn in a world and be able to enter the nether itself. Given that we were able to achieve the building of the portal, our current objective is to get the agent to navigate and build the portal from a set seed. 

We chose to focus on a set seed as we think it would be simpler to allow it to converge on a "best" solution, but we plan to attempt random seeds as well later in the project.

## Approach

Using reinforcement learning, our agent is learned how to build, light, and enter a nether portal. Now, we are training it to successfully achieve this on a set seed.

## Evaluation

Currently, we are evaluating our agents on time, the faster the time, the better the agent. Using rewards, we are guiding our agents towards the correct path on the set seed.

## Remaining Goals and Challenges

We still plan to train our agent to enter the nether on random seeds, and if there's time, complete speedruns. However, this will still take a lot of time and we are unsure if we will be able to achieve this.

## Resources Used

    Our main repository can be found at this link: 
[Main Repo] (https://github.com/freakmykappachunguslife/MCSRAI)

The main tools we used was MineRL and MineDojo, both created for Minecraft AI:
[MineDojo] (https://minedojo.org)

[MineRL] (https://minerl.readthedocs.io/en/latest/index.html)


## Video Summary

Below is the video of our best agent completing the nether portal, given the obsidian and flint and steel.

<video controls src="videos/buildportal.mp4" type="video/mp4" width="600" height="400">
    Your browser does not support the video tag.
</video>

Since we were able to achieve this goal, we have begun training an agent to gather the resources.

Here's a run with completely randomized inputs (beware there are flashing screens in this video):

<video controls src="videos/randomagent.mp4" type="video/mp4" width="600" height="400">
    Your browser does not support the video tag.
</video>

You can see that the agent does manage to get seeds, but that is simply random chance. The agent does a whole lot of nothing. The most glaring issue with the random agent is that it is, well, random; often times it opens the inventory, and proceeds to randomly press buttons and moving the mouse, which does nothing for approaching the end goal. For this reason, we heavily punished the agent for opening the inventory.
