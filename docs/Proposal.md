## 2.1

## 2.2 Summary of Project
The TAS, or tool-assisted-speedrun, is a sub-category of speedrunning where a computer with predetermined inputs, plays the game perfectly, achieving runs with zero error. With the TAS as inspiration, we wanted to see if we could create an AI that speedruns Minecraft. To achieve this, our AI agent would perform certain keystrokes or mouse inputs as a result of its coordinate position and current frame. 

## 2.3 Project Goals
Our most basic goal is for our agent to build a nether portal, but with the materials already given to it. The agent making the proper keystrokes would be a good proof of concept at the minimum. More realistically, we want the agent to be able to build a nether portal from a fresh spawn on a set seed (for replicability and ease). This would allow us to see how complex we can manage to create our model. Ideally our agent would beat the game, but that is much too out of scope for this project and our feeble capabilities. So, our moonshot goal would be to achieve the nether portal on a random seed, adding the difficulty of randomness.

## 2.4 AI/ML Algorithms

## 2.5 Evaluation Plan
To evaluate our Minecraft agent, we’re going to have it run several times on a fixed Minecraft seed within a fixed amount of time. The most basic metric will be how fast our agent can achieve the task (if at all). For the minimum goal of building a nether portal, the metrics will be the percentage of runs where the Minecraft agent is able to successfully build the nether portal out of N total runs. For the realistic goal, we measure the time it takes for the Minecraft agent to achieve the required materials to build a nether portal; wood → wooden axe → stone pickaxe →  iron →  water bucket →  lava →  obsidian. The baselines will include setting up a simple random agent that randomly selects an action, a rule based agent where the instructions and actions are hardcoded, and finally our Minecraft RL Agent. For all 3 models, we’ll compare each model's ability to build nether portals out of N times, and whether or not the models are able to achieve the required materials.

For our qualitative analysis, we’ll verify that our project works by making sure that our Minecraft AI agent is able to perform basic behaviors such as placing blocks, mining wood and iron, crafting the tools in the correct order, and building  a nether portal with a water bucket by placing it on top of lava. For each of these basic behaviors, we’ll create a controlled environment where the Minecraft AI agent will solely be graded on whether or not it can perform that behavior. By creating controlled environments for each behavior, we can see where our Minecraft Agent is able to perform and the areas in which it needs to improve.
## 2.6 AI Usage