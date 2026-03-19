---
layout: default
title: Final Report
---

# Create New World:

## Getting Started

MCSRAI began as an idea: create an AI model that is able to speedrun Minecraft. During COVID, with
nothing to do, many people took to speedrunning Minecraft, reviving the interest. And recently, with
the introduction of a mod, "MCSR: Ranked", the hobby/event has blown up once more. Allowing people
to get a physical rank attached to their performance, along with large content creators attempting
runs themselves, Minecraft Speedrunning is reaching its highest peak since the pandemic.

The concept of creating an AI model to speedrun Minecraft was enticing to say the least, and that is
where we began MCSRAI.

Soon after getting the idea, we began to think about how we would train the model. Speedrunning is
quite complex, with many major milestones to achieve, all while making many micro-decisions and making
precise calculations in real-time.

We broke down speedrunning into these major milestones:

- Food and Bucket
- Enter Nether
- Bastion
- Ender Pearls
- Fortress and Blaze
- Exiting Nether and Triangulating Stronghold
- Find End Portal
- Kill Dragon & Complete Speedrun

With each of these major milestones, we figured that this would be too much for our knowledge and
capability. As such, we needed to eliminate milestones which were too challenging, such as killing the
dragon, blazes, and the bastion. These tasks relied on the NPC's algorithm, which would potentially
create confounding variables in our training. Along with these "combat" milestones, we elimiated the
major navigation-based tasks, being finding the bastion, fortress, and stronghold. We decided the
nuances of finding these structures would be beyond our skill level.

This left us with two milestones: "Food and bucket" and "Enter Nether".

We decided on the most important/iconic of the milestones: Entering the Nether.
It's a massive milestone in any speedrun, usually being the first "split" of the run and
marking the potential for any good run.

## Taking Inventory

First, we had to decide our goals, and as prompted by the first report, we chose our baseline goal to just create the portal frame. Our target would be to perform a nether portal run on a set seed and the reach goal would be to achieve this on random seeds.

So, we set out to find any tools for training models regarding Minecraft. This led us to MineRL, an open source RL tool for Minecraft. Utilizing "MalMo", Microsoft's tool for AIs in Minecraft, MineRL allows the creation of a Minecraft environment in which the agent can observe, modify, and, well, play Minecraft!

However, MineRL is sligtly outdated, but luckily, it led us to a more updated tool, MineDojo. This quickly became the backbone of our entire project. As a more modern tool, it allowed us to run more efficiently, observe our model's performance better, and alter more variables.

# A Quick Intro to a MineDojo Environment

The MineDojo environment is a Minecraft world in which the Malmo agent is able to run and input commands. The choices the agent can take is the "action space", with basic movement inputs like "WASD" and "Spacebar". This also includes item management inputs like "1-9" and "E". Furthermore this also includes camera movement, allowing the agent to move the head left, right, up, and down. The observation space is two main parts: the voxel space and the RGB pixel frame. The voxel space is a 9x9x9 block area around the agent, denoting each Minecraft block within each tile in this space. The RGB pixel frame is a 2D array, with each entry being the RGB value of the pixel at that location (x,y) in the frame of the display.

## Stone Age

As we began to train, we noticed issues. First off, we needed to redifine our "start state". With the default environment, the terrain wasn't flat. This could cause our agent to create "bad" portals, despite it thinking it was (attached video is example ADD TS FROM BEFORE).

So, we changed our "start state" to begin on a superflat world. This would create completely flat and replicable terrain for our agent to train on. But with this came the fact that there is no method to create a portal in this superflat world. So, we gave the agent the nescessary resources to begin building.

Initially, we looked at the base choice: random. Take a random input from the action space, and do it. Do this over 10000 timesteps, and see what happens. As expected, this model was terrible! It did a whole lot of nothing, tossing items around, opening the inventory endlessly, and other inefficient inputs. And as such, we decided to redefine the action space.

We rid ourselves of many uneventful actions, like "Q", "E", "F", and "ESC" for example. This would make our agent to always pick some input that guided us towards completion.

With these basic things in mind, we were able to begin really training our model.

# Approach and Evaluation:

## Aquire Hardware

With our redefined start states, we could begin to train this model, primarily using PPO.

By rewarding the model for placing obsidian, our model learned to place obsidian.

- Reward: Place obsidian; +1

By punishing the model for taking damage, we prevented the model from lighting itself with the flint and steel (and avoid dying!)

- Punishment: Take damage; -0.1

However, as we trained, we observed the model learned to place all the obsidian, although haphazardly.

We were making progress, avoiding deaths from fire and placing obsidian, but the models were not close to creating a valid portal frame. Despite this, we were moving towards our goal and this was a good first step.

# Ice Bucket Challenge

So, to avoid the model from sporadically placing obsidian, we added more rewards and punishments. To prohibit "bad" placement of obsidian, we punished the model for each obsidian that did not conform to a valid frame, checking using the voxel space. But, to encourage valid frames, we rewarded for correct location of obsidian. Furthermore, we also added a punishment for each timestep during training, which we used to push the model to choose actions earlier.

- Reward: Proper frame location; +2
- Punishment: Improper frame location; -1 | Timestep ; -0.01

This led us to the following graph:

INSERT HERE

** A Preface: All the future graphs will have the total reward value in the negatives. This is expected and due to the punishment for timesteps **

As we observed, the model began very poorly, still haphazardly placing obsidian; the old model accumulated many punishments. But as we continued to train the model, we observed our scores improving!

This was exciting to observe on Tensorboard, seeing the total reward begin to rise! We had began to create a "good" model!

Or so we thought.

## Hot Stuff

Pike syndrome is a term created by behavioural biologists. They placed an aggressive Pike in a tank with multiple prey fish, but with a glass wall between them. The Pike, seeing prey, darted towards the fish, but slammed into the glass. Repeatedly. Eventually, the Pike became docile. And even with the glass removed the pike would not attempt to eat the prey fish. Even swimming by the pike's mouth would not garner a reaction.

Why is this relevant to our model?

It's because we gave our model Pike syndrome.

As our inital model was terrible, it accumulated tons of punishment. So, as we trained it on the new reward/punishment system,iIt learned to never place "bad obsidian" by simply not placing any obsidian. This way, it never accumulated its previous punishments and the reasoning behind our seemingly improving scores.

It learned to not do anything. By mucking around, the model would be able to "maximise" its reward by reducing its punishments.

# We Need to Go Deeper

This issue with our model's "learned helplessness" is attributed to our rewards. We thought these were logical choices for reward/punishment, however, we were not thinking like a computer program, we were thinking like humans.

So, we tried something a little more discrete rather than continuous.

By dynamically changing our reward system during the training, we could attempt to reinforce good behaviour at a certain step and later reduce bad behavior.

This led us to a new system:

## Phase 1: Any Obsidian is Good Obsidian (0–100k steps)

In this phase, the model receives +0.5 for every obsidian block placed, regardless of where it lands. No penalties. No geometry checks. Just a simple, unambiguous signal: placing blocks is good.

## Phase 2: Not Just Anywhere (100k–200k steps)

With the model now committed to placing blocks, we introduced a soft penalty of -0.5 for any obsidian placed outside a valid frame position. Critically, the +0.5 flat reward remains active — a misplaced block still earns something, it just earns less than a correctly placed one. The model is never punished into inaction; it is nudged toward better choices while still being rewarded for trying.
This phase acts as a transition. The model begins to notice that some placements feel better than others, without ever hitting the sharp negative signal that caused learned helplessness in the first place

## Phase 3: The Frame (200k+ steps)

In the final phase, the full reward signal activates. Blocks placed on one of the 14 valid nether portal frame positions earn an additional +2.0 on top of the flat reward, making correct placement worth +2.5 total versus +0.5 for a misplaced block. The model now has a clear target and the training history to pursue it without collapsing back into passivity.

# Sucess?

Not Really, : put in video here

# A Different Approach

By this point, we noticed our previous model had basically learned the wrong lesson. Instead of learning how to build a portal, it had learned to spin around wildly and place obsidian wherever it could. It was getting reward for placing blocks, but it was not learning the structure of a valid frame. So while it looked like we were making progress, the model was really just getting better at random placement.
<video controls src="videos/ppo_obsidian-step-9000-to-step-9500.mp4" type="video/mp4" width="600" height="400">
Your browser does not support the video tag.
</video>

At this point, we started to think that the issue was not just the reward system, but also the way the model was allowed to move. Even after simplifying the action space, the camera movement was still far too messy. The agent would look in strange directions, turn too much, and lose track of where it had already placed obsidian. For a task like portal building, that was a huge problem.

So, we changed the action space again. Rather than letting the model control random low level camera movement, we switched to a smaller set of discrete actions. Instead of trying to learn every possible pitch and yaw movement, the model could now choose simple actions like look left, look right, look up, look down, move, place obsidian, jump and place obsidian, and ignite the portal. This made the camera much smoother and made the task much easier to learn.
<video controls src="videos/RnadomlyPlacingBlocks.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

From there, we also split the task into phases. First came the build phase. In this phase, the model only had to make the obsidian frame. If it tried to use the flint and steel too early, that action would be blocked and punished. Then came the light phase. Once the frame was complete, the model was encouraged to switch to the flint and steel and ignite the portal. Successfully lighting the portal gave a large reward of +40.

Even with this change, the model still had trouble finishing the full frame consistently. It would often start well, but then drift away and place blocks in bad positions. Still, by around 50k timesteps, we could see that the model had at least learned one useful behavior: it was now intentionally placing obsidian blocks rather than just wandering around aimlessly.
<video controls src="videos/differentapproach50k.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

So, we broke the build phase down even further into smaller subgoals. Completing the bottom row gave a reward of +4. Completing the left side gave +3. Completing the right side gave +3. Completing the top row gave +4. The goal here was to stop treating the portal like one big all or nothing task, and instead reward the model for making progress piece by piece.

But this led to another issue. The model seemed to figure out that the easiest reward to exploit was the bottom row reward. Rather than building the sides and finishing the portal, it kept repeating the bottom row pattern again and again. Instead of making a frame, it would place four obsidian in a row, then another four, then another four, creating a long line of obsidian across the ground. By around 200k timesteps, the model looked much more deliberate than before, but it was still exploiting this bottom row behavior rather than truly completing the portal.
<video controls src="videos/differentapproach200k.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

It is possible that this was partly a programming mistake on our end. Our bottom row check may have been too generous, or the reward may have triggered in situations we did not fully intend. But more importantly, it showed us a much bigger lesson about reinforcement learning. Even when a reward system feels logical to us, the model may still find a loophole that maximizes reward without solving the actual task.

# Oversights

Luckily, we learned early in the project that the goal isn't to create a working model, although it is best that you can, but to learn from this experience.

The main things were learned form this experience were the following:

-

In future, we could/should implement the following:

- Reducing the size of the RGB frame for faster computation (the model does not need to focus on the extraneous noise on the side of the frame); we could do this by completely culling the values towards the edge of the image, allowing the model to focus on what's ahead of it.
- Use a better tool; MineDojo and MineRL both only run on 1.11, which is a older version of Minecraft and not the common version players speedrun on, which is post 1.16.1.

## Resources Used:

## Video Walkthrough:

# CITE SOURCES REMEMBER
