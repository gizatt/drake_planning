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
from pydrake.trajectories import (
    PiecewisePolynomial, PiecewiseQuaternionSlerp)

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
        RobotPlanRunner(is_discrete=True, control_period_sec=1e-3))
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
    def MakeQPlanData():
        t_knots = np.array([0., 2., 4.])
        q_knots = np.array([
            [0., 0.6, 0., -1.75, 0., 1., 0.],
            [0.1, 0.6, 0., -1.75, 0., 1., 0.],
            [-0.1, 0.6, 0., -1.75, 0., 1., 0.]]).T
        q_traj = PiecewisePolynomial.Cubic(
            t_knots, q_knots, np.zeros(7), np.zeros((7)))
        return PlanData(PlanType.kJointSpacePlan,
                          joint_traj=q_traj)
    def MakeEEPlanData():
        t_knots = np.array([0., 2., 4.])
        ee_xyz_knots = np.array([
            [0.7, 0., 0.1],
            [0.7, 0.2, 0.3],
            [0.7, -0.2, 0.1]]).T
        ee_quat_knots = [
            RollPitchYaw(0., np.pi, 0).ToQuaternion(),
            RollPitchYaw(0., np.pi, 0).ToQuaternion(),
            RollPitchYaw(0., np.pi, 0).ToQuaternion()
            ]
        ee_xyz_traj = PiecewisePolynomial.FirstOrderHold(
            t_knots, ee_xyz_knots)
        ee_quat_traj = PiecewiseQuaternionSlerp(
            t_knots, ee_quat_knots)
        return PlanData(PlanType.kTaskSpacePlan,
                        ee_data=PlanData.EeData(
                            p_ToQ_T=np.zeros(3),
                            ee_xyz_traj=ee_xyz_traj,
                            ee_quat_traj=ee_quat_traj))

    q_plan = MakeQPlanData()
    ee_plan = MakeEEPlanData()

    plan_sender = builder.AddSystem(PlanSender([q_plan, ee_plan]))
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
    simulator.AdvanceTo(plan_sender.get_all_plans_duration())


if __name__ == "__main__":
    main()
