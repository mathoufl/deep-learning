# reward 
whenever the agent take damage, it receive a reward equal to -damage_taken

# map list
The map are all stored inside the .wad
they can be chosen with ```game.set_doom_map(string : map name)``` 

## MAP01
5 imps and a shotgun that spawn at random at one of two locations
\-> the agent should learn the value of picking up the shotgun

## MAP02
4 cacodemon (strong enemies), and a pillar that can be used as cover. The agent start with a supershotgun
\-> the agent should learn to dodge projectiles

