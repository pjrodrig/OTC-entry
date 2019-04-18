## Research
- Neural networks
- Deep learning
- Merging Neural Networks



## AI
Goals
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
