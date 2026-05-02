## PWHL WAR Project
The goal of this project was to produce a Wins Above Replacement model for the PWHL.
In order to properly calculate this, I took an approach of combing Offensive WAR (oWAR) and Defensive WAR (dWAR).
The primary input into oWAR is an expected goals model based on shot coordinates this with some weighting around +/- gives us our oWAR.
dWAR is tricker, without coordinate level data on the defensive end it is mostly an aggregate of typical defensive stats.

### Expected Goals
Now for my presentation project for class I focused in on Expected Goals. 
The question that I wanted to answer is can we predict how many goals a player should score? And will that be better than randomly guessing. 
We get the player level data from hockey-statistics.com and the shot coordinate data from pwhl.hockey-statistics.com
The main findings were that we can in fact predict goals and have it be better than randomly guessing. With only 2 seasons of training data there isn't much more we can say than that. 
With more seasons in the data set we could move towards minimizing the error which is currently around 1.4 goals per player.
