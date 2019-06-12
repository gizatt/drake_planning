import argparse
import os
import sys

import matplotlib.pyplot as plt
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
from pydrake.systems.framework import (
    AbstractValue, BasicVector, DiagramBuilder, LeafSystem)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter
from pydrake.systems.sensors import (
    Image,
    PixelFormat,
    PixelType
)
from pydrake.trajectories import (
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp
)


class PrimitiveDetectionSystem(LeafSystem):
    ''' Consumes robot state and checks primitive status against it
    periodically, publishing to console. '''
    def __init__(self, mbp, grab_period=0.1):
        LeafSystem.__init__(self)

        self.mbp = mbp
        # Our version of MBP context, which we'll modify
        # in the publish method.
        self.mbp_context = mbp.CreateDefaultContext()

        # Object body names we care about
        self.body_names = ["base_link"]

        self.set_name('primitive_detection_system')
        self.DeclarePeriodicPublish(grab_period, 0.0)
        
        # Take robot state vector as input.
        prototype_rgb_image = Image[PixelType.kRgba8U](0, 0)
        prototype_depth_image = Image[PixelType.kDepth16U](0, 0)
        self.DeclareVectorInputPort("mbp_state_vector",
                                    BasicVector(mbp.num_positions() +
                                                mbp.num_velocities()))

    def DoPublish(self, context, event):
        # TODO(russt): Change this to declare a periodic event with a
        # callback instead of overriding DoPublish, pending #9992.
        LeafSystem.DoPublish(self, context, event)
        print("Curr sim time: ", context.get_time())

        mbp_state_vector = self.EvalVectorInput(context, 0).get_value()
        self.mbp.SetPositionsAndVelocities(self.mbp_context, mbp_state_vector)

        # Get pose of object
        for body_name in self.body_names:
            print(body_name, ": ")
            print(self.mbp.EvalBodyPoseInWorld(
                self.mbp_context, self.mbp.GetBodyByName(body_name)).matrix())


class CameraCaptureSystem(LeafSystem):
    ''' Example system that periodically
    grabs RGB-D image inputs. If given matplotlib axes,
    draws the RGB and Depth images to them when they're grabbed. '''
    def __init__(self, grab_period=0.1, ax_rgb=None, ax_depth=None):
        LeafSystem.__init__(self)

        self.set_name('camera_capture_system')
        self.DeclarePeriodicPublish(grab_period, 0.0)
        # RGB and D image input ports.
        # Declaring input ports requires supplying a prototype
        # of the input type.
        prototype_rgb_image = Image[PixelType.kRgba8U](0, 0)
        prototype_depth_image = Image[PixelType.kDepth16U](0, 0)
        self.DeclareAbstractInputPort("rgb_image",
                                      AbstractValue.Make(
                                        prototype_rgb_image))
        self.DeclareAbstractInputPort("depth_image",
                                      AbstractValue.Make(
                                        prototype_depth_image))

        self.rgb_data = None
        self.depth_data = None
        if ax_rgb is not None:
            self.rgb_data = ax_rgb.imshow(np.zeros([1, 1, 4]))
        if ax_depth is not None:
            self.depth_data = ax_depth.imshow(np.zeros([1, 1]))

    def DoPublish(self, context, event):
        # TODO(russt): Change this to declare a periodic event with a
        # callback instead of overriding DoPublish, pending #9992.
        LeafSystem.DoPublish(self, context, event)
        print("Curr sim time: ", context.get_time())

        rgb_image = self.EvalAbstractInput(context, 0).get_value()
        depth_image = self.EvalAbstractInput(context, 0).get_value()

        if self.rgb_data is not None:
            self.rgb_data.set_data(rgb_image.data)
            plt.gcf().canvas.draw()
        if self.depth_data is not None:
            self.depth_data.set_data(depth_image.data.squeeze())
            plt.gcf().canvas.draw()


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

    # Hook up cameras
    primitive_detection_system = builder.AddSystem(
        PrimitiveDetectionSystem(
            station.get_multibody_plant()))
    builder.Connect(
        station.GetOutputPort("plant_continuous_state"),
        primitive_detection_system.GetInputPort("mbp_state_vector"))

    #fig = plt.figure()
    #fig.show()
    #camera_capture_systems = []
    #camera_names = station.get_camera_names()
    #for cam_i, name in enumerate(camera_names):
    #    ax_rgb = plt.subplot(len(camera_names), 2, cam_i*2 + 1)
    #    ax_depth = plt.subplot(len(camera_names), 2, cam_i*2 + 2)
    #    camera_capture_system = builder.AddSystem(CameraCaptureSystem(
    #        grab_period=1.0,
    #        ax_rgb=ax_rgb,
    #        ax_depth=ax_depth))
    #    camera_capture_system.set_name("capture_%d" % cam_i)
    #    builder.Connect(station.GetOutputPort("camera_%s_rgb_image" % name),
    #                    camera_capture_system.GetInputPort("rgb_image"))
    #    builder.Connect(station.GetOutputPort("camera_%s_depth_image" % name),
    #                    camera_capture_system.GetInputPort("depth_image"))
    #    camera_capture_systems.append(camera_capture_system)

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
