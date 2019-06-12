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
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    AbstractValue, BasicVector, DiagramBuilder, LeafSystem)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter, ConstantVectorSource
from pydrake.systems.sensors import (
    Image,
    PixelFormat,
    PixelType
)
from pydrake.trajectories import (
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp
)

from underactuated.planar_scenegraph_visualizer import PlanarSceneGraphVisualizer
from mouse_keyboard_teleop import MouseKeyboardTeleop, print_instructions
from differential_ik import DifferentialIK


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
    mbp = station.get_multibody_plant()
    station.SetupDefaultStation()
    station.Finalize()
    iiwa_q0 = np.array([0.0, 0.6, 0.0, -1.75, 0., 1., 0.])

    use_plan_runner = False
    if use_plan_runner:
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
        end_time = plan_sender.get_all_plans_duration()
    else: # Hook up DifferentialIK
        robot = station.get_controller_plant()
        params = DifferentialInverseKinematicsParameters(robot.num_positions(),
                                                         robot.num_velocities())

        time_step = 0.005
        params.set_timestep(time_step)
        # True velocity limits for the IIWA14 (in rad, rounded down to the first
        # decimal)
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        # Stay within a small fraction of those limits for this teleop demo.
        factor = 1.0
        params.set_joint_velocity_limits((-factor*iiwa14_velocity_limits,
                                          factor*iiwa14_velocity_limits))

        differential_ik = builder.AddSystem(DifferentialIK(
            robot, robot.GetFrameByName("iiwa_link_7"), params, time_step))
        differential_ik.parameters.set_nominal_joint_position(iiwa_q0)

        builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                        station.GetInputPort("iiwa_position"))

        print_instructions()
        teleop = builder.AddSystem(MouseKeyboardTeleop(grab_focus=True))
        filter_ = builder.AddSystem(
            FirstOrderLowPassFilter(time_constant=0.005, size=6))

        builder.Connect(teleop.get_output_port(0), filter_.get_input_port(0))
        builder.Connect(filter_.get_output_port(0),
                        differential_ik.GetInputPort("rpy_xyz_desired"))

        builder.Connect(teleop.GetOutputPort("position"), station.GetInputPort(
            "wsg_position"))
        builder.Connect(teleop.GetOutputPort("force_limit"),
                        station.GetInputPort("wsg_force_limit"))

        fft = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
        builder.Connect(fft.get_output_port(0), 
                        station.GetInputPort("iiwa_feedforward_torque"))
        end_time = 10000

    # Attach a visualizer.
    meshcat = False # TODO(gizatt) Add to argparse
    if (meshcat):
        meshcat = builder.AddSystem(MeshcatVisualizer(
            station.get_scene_graph(), zmq_url=args.meshcat))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))
    else:
        plt.figure()
        plt.gca().clear()
        viz = builder.AddSystem(PlanarSceneGraphVisualizer(
            station.get_scene_graph(),
            xlim=[0.25, 0.8], ylim=[-0.1, 0.5],
            ax=plt.gca()))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        viz.get_input_port(0))

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

    #station.GetInputPort("wsg_force_limit").FixValue(station_context, 40.0)
    #station.GetInputPort("wsg_position").FixValue(station_context, 0.0)
    differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
        differential_ik, diagram_context), iiwa_q0)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(end_time)


if __name__ == "__main__":
    main()
