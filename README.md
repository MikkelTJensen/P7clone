## 7th Semester Code Clone

## P7-Code

This is the code submitted by cs-21-dat-7-03 for the 7th semester on Computer Science at Aalborg University.
A paper was written and submitted for the scalability results of the developed DQNPER agents.
These agents were developed to control power grids in the L2RPN challenge.
Agents are developed using the Grid2Op framework.
The code base additionally facilitates training and comparing different agents.

## Setup
This could be setup in a virtual environment but does not need to.
```
pip install -r requirements.txt
```

## Train and Evaluate
The experiment threeAgents can be trained and evaluated with the following command, and can be seen in the experiment folder.
```
python3 -W ignore runner.py --f threeAgents --t --e
```

## Compare
After the agents have been trained, they can be compared, where they get scored in the L2RPN challenge.
```
python3 comparer.py --f threeAgents --a --c
```
The scores and the analysis of the actions are saved in the comparison results folder of the agents inside the experiment folder.

## Plot
A script for plotting was also developed. The script can be modified to also fetch average agent scores.
```
python3 plotter.py
```