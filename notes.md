## Challenge resources
- [blog](https://blogs.unity3d.com/2019/01/28/obstacle-tower-challenge-test-the-limits-of-intelligence-systems/)
- [info](https://www.aicrowd.com/challenges/unity-obstacle-tower-challenge)
- [challenge repo](https://github.com/Unity-Technologies/obstacle-tower-challenge)
- [paper](https://storage.googleapis.com/obstacle-tower-build/Obstacle_Tower_Paper_Final.pdf)
- [env repo](https://github.com/Unity-Technologies/obstacle-tower-env)
- [rules](https://gitlab.aicrowd.com/unity/obstacle-tower-challenge-resources/blob/master/Rules.md?_ga=2.166169982.1364833215.1555537393-735101082.1555262287) 
- [leaderboard](https://www.aicrowd.com/challenges/unity-obstacle-tower-challenge/leaderboards)
- [submission]()

## Research
- Neural networks
    - https://www.3blue1brown.com/videos
- stochastic gradient descent
    - https://github.com/mnielsen/neural-networks-and-deep-learning
    - https://en.wikipedia.org/wiki/Stochastic_gradient_descent
- assigning initial values
    - gaussian distribution
- Deep learning
- Dynamically expandable neural networks
    - https://hackernoon.com/dynamically-expandable-neural-networks-ce75ff2b69cf
- Backpropagation
    - http://colah.github.io/posts/2015-08-Backprop/
- semantic segmentation


## AI
Goals
- detect surfaces
- identify inputs
- identify types of actions
    - types
        - flat directional movment
            - left
            - right
            - forward
            - backward
            - forward-left
            - forward-right
            - backward-left
            - backward-right
        - vertical movment
            - jump
        - camera movement
            - cam left
            - cam right
    - can be initially solved by a human
- identify goals for types of actions
    - flat directional movement
        - destination
    - vertical movement
        - overcome impassible terrain
        - reach floating items
    - camera movement
        - map room
        - find goals
- solve problems individually
    - flat directional movment & 
        - when can an action be taken/what actions influence change
- identify corrilations
    - mapped room -- traversible terrain



slice tower into more easily solved problems

  the horizontal movement problem can be converted into a top down arcade like game where we must move a model around a 2d environment. Impassible terrain must be identified to draw a map of the 2d environment. gaps that must be jumped across should be represented in such a way that we allow our bot to traverse over gaps that are jumpable

  to solve the vertical movement / jump problem, we must identify linear paths from the horizontal movement problem and display them as a side scrolling platformer



Actions
```
0 = none----------|--|--|--|
1 = right---------|  |  |  |
2 = left----------|  |  |  |
3 = jump-------------|  |  |
6 = cam left------------|  |
12 = cam right----------|  |
18 = forward---------------|
36 = backward--------------|
```
action = sum of choices


detect similar colors, look for shapes in groupings



## Human Comparison

### Brain Functions: 

- sight
    - identify objects
    - identify position changes
    - identify external movement

- movement
    - move to goal
    - jump
        - jump over obstacles
        - jump to platforms

- time
  - recognize time it will take to get to a goal
  - recognize timing of external movement



### Questions for AI to answer
- What does the room look like?
- Has anything changed?
- Has progress been made?
- Will choosing the action have an effect? (can take action, ex. jumping while in the air, going forward against a wall)
- what type of room is the ai in
- what state is the ai in? does it have a key? 


## Tower Notes

Types of nodes:
- Start (S): is the starting node in the dungeon.
- Exit (E): is the end node in the dungeon that the player
need to reach.
- Key (K): is a node that contains a key.
- Lock (L): is a node that can’t be entered without a key.
- Puzzle (P): is a node that contain a puzzle/challenge that
need to be passed to continue.
- Lever (V): is a node that contains a lever that need to be
pressed to pass to the next room.
- Normal (N): any other node that doesn’t follow any of the
above.