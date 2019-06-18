# Executing abstracted tasks with state machines and Pydrake

Provides some systems that load a state machine (compiled from e.g. [the slugs GR(1) synthesizer](https://github.com/VerifiableRobotics/slugs) and deploy it in a Drake simulation of a robot.

[![Example video](https://img.youtube.com/vi/ZyfcRW-SiRY/0.jpg)](https://www.youtube.com/watch?v=ZyfcRW-SiRY)

## Setup and usage
Tested on Python 2.7. Should hopefully work on Python 3 as well soon.

1) Install Drake binaries for use with Python (at least as recent as 2019 June 17) from [here](https://drake.mit.edu/python_bindings.html).
2) Clone the [MIT underactuated class repository](https://github.com/RussTedrake/underactuated) and add `underactuated/src` to your `PYTHONPATH`. (This repo provides a few extra useful Drake tools that we use.)
3) Invoke the example with `python run_manipulation_station_pick_and_place.py`. A matplotlib window should open showing the robot doing cool things!

## Changing the specification

1) Install and build ([the slugs GR(1) synthesizer](https://github.com/VerifiableRobotics/slugs), and add its root path to your environment as `SLUGS_DIR`. e.g. `export SLUGS_DIR=/home/gizatt/tools/slugs`.
2) Change the specification `*.structuredslugs` file (and possibly the compiler helper script) in the `specifications` folder. Easy first modifications are to change the `SYS_LIVENESS` constraints to change the behaviors that the robot should execute. (See the slugs docs for details, but in short, each of those conditions needs to be achieved in order, repeatedly forever.)

## File overview:
- *run_manipulation_station_pick_and_place.py*: Entrypoint, run this with no arguments to see things in motion. (It should open a matplotlib window and show the robot moving blocks around, see above video.) Run with `--teleop` to control the robot manually with mouse and keyboard (see terminal for keybinds).
- *specifications/\**:
    - *\*.structuredslugs*: Specification file defining the consequences of primitives and the desired abstract behavior.
    - *\*.json*: State machines, compiled from GR(1) specs to accomplish a task.
    - *\*.sh*: Helper script to compile the specification file into a state machine.
- *primitives.py*: Defines Primitives, which know how to take an initial state of the robot and, if the primitive is valid, produce a trajectory to execute an action.
- *symbol_map.py*: Defines Symbols, which take a state of the robot and return a (hopefully meaningful) boolean value.
- *differential_ik.py*: Provides the robot with a controller to track end effector trajectories. (TODO: grab this from Drake / replace with PlanRunner.)
- *mouse_keyboard_teleop.py*: Utilities / controller for the teleop mode.
