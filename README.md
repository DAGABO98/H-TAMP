# H-TAMP
Repo for Heterogeneous Task and Motion Planning

## Commands

To generate traversal graph:
```console
python -m HTAMP.environment.traversal_graph_gen
```

To generate occupancy reservations:
```console
python -m HTAMP.environment.grid_world
```

To test motion planning:
```console
python -m HTAMP.planning.motion_planner --use_saved_data --num_robots 5
```

To test movement visualization:
```console
python -m HTAMP.planning.state --use_saved_data --num_robots 5
```
