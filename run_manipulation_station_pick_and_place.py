import argparse
import os
import sys

import numpy as np


from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface,
    CreateDefaultYcbObjectList)
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.multibody.plant import MultibodyPlant
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.manipulation.robot_plan_runner import (
    PlanData,
    PlanSender,
    PlanType,
    RobotPlanRunner
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (BasicVector, DiagramBuilder,
                                       LeafSystem)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter
from pydrake.trajectories import PiecewisePolynomial

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    builder = DiagramBuilder()

    # Set up the ManipulationStation
    station = builder.AddSystem(ManipulationStation())
    station.SetupDefaultStation()
    station.Finalize()

    # Set up the PlanRunner
    plan_runner = builder.AddSystem(
        RobotPlanRunner(is_discrete=False, control_period_sec=1e-3))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    plan_runner.GetInputPort("iiwa_position_measured"))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    plan_runner.GetInputPort("iiwa_velocity_estimated"))
    builder.Connect(station.GetOutputPort("iiwa_torque_external"),
                    plan_runner.GetInputPort("iiwa_torque_external"))
    builder.Connect(plan_runner.GetOutputPort("iiwa_position_command"),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(plan_runner.GetOutputPort("iiwa_torque_command"),
                    station.GetInputPort("iiwa_feedforward_torque"))

    # Set up a simple PlanSender that sends multiple
    # plans in sequence
    t_knots = np.array([0., 1., 2.])
    q_knots = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [1, 1, 1, 1, 1, 1, 1]
    ]).T
    traj = PiecewisePolynomial.Cubic(
        t_knots, q_knots, np.zeros(7), np.zeros((7)))
    plan_1 = PlanData(PlanType.kJointSpacePlan, traj)
    plan_sender = builder.AddSystem(PlanSender([plan_1]))
    builder.Connect(plan_sender.GetOutputPort("plan_data"),
                    plan_runner.GetInputPort("plan_data"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    plan_sender.GetInputPort("q"))

    # Attach a visualizer.
    meshcat = builder.AddSystem(MeshcatVisualizer(
        station.get_scene_graph(), zmq_url=args.meshcat))
    builder.Connect(station.GetOutputPort("pose_bundle"),
                    meshcat.get_input_port(0))

    # Remaining input ports need to be tied up.
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    station_context = diagram.GetMutableSubsystemContext(
        station, diagram_context)

    station.GetInputPort("wsg_force_limit").FixValue(station_context, 40.0)
    station.GetInputPort("wsg_position").FixValue(station_context, 0.0)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(10.0)


if __name__ == "__main__":
    main()
