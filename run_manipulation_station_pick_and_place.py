import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface,
    CreateDefaultYcbObjectList)
from pydrake.geometry import (
    Box,
)
from pydrake.multibody.tree import (
    PrismaticJoint,
    SpatialInertia,
    RevoluteJoint,
    UniformGravityFieldElement,
    UnitInertia,
    FixedOffsetFrame
)
from pydrake.multibody.plant import (
    MultibodyPlant, CoulombFriction)
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
from pydrake.systems.primitives import (
    FirstOrderLowPassFilter,
    ConstantVectorSource,
    TrajectorySource
)

from underactuated.planar_scenegraph_visualizer import PlanarSceneGraphVisualizer
from mouse_keyboard_teleop import MouseKeyboardTeleop, print_instructions
from differential_ik import DifferentialIK

from symbol_map import *
from primitives import *


class TaskExectionSystem(LeafSystem):
    ''' Given a compiled JSON DFA and lists of symbol and 
    primitive objects with matching names, '''

    def __init__(self, mbp, symbol_list, primitive_list, dfa_json_file, update_period=0.05):
        LeafSystem.__init__(self)

        self.mbp = mbp
        # Our version of MBP context, which we'll modify
        # in the publish method.
        self.mbp_context = mbp.CreateDefaultContext()

        self.set_name('task_execution_system')

        # Take robot state vector as input.
        self.DeclareVectorInputPort("mbp_state_vector",
                                    BasicVector(mbp.num_positions() +
                                                mbp.num_velocities()))

        # Load the JSON file
        with open(dfa_json_file, 'r') as f:
            json_data = json.load(f)

        # Figure out which states in the JSON dict are environment
        # symbols, and which are primitives that we can execute.
        environment_symbol_indices = []
        environment_symbols = []
        action_primitive_indices = []
        action_primitives = []

        for var_i, var_name in enumerate(json_data["variables"]):
            # Split into symbols and primitives.
            found_symbol = None
            found_primitive = None
            for symbol in symbol_list:
                if symbol.name == var_name:
                    if found_symbol is not None:
                        raise ValueError("Ambiguous matching symbol names provided for symbol {}.".format(var_name))
                    found_symbol = symbol
            for primitive in primitive_list:
                if primitive.name == var_name:
                    if found_primitive is not None:
                        raise ValueError("Ambiguous matching primitive names provided for symbol {}.".format(var_name))
                    found_primitive = primitive
            if found_primitive and found_symbol:
                raise ValueError("Both matching symbol AND primitive found for symbol {}.".format(var_name))
            if not found_primitive and not found_symbol:
                raise ValueError("No matching symbol or primitive found for symbol {}.".format(var_name))
            
            if found_symbol is not None:
                environment_symbol_indices.append(var_i)
                environment_symbols.append(found_symbol)
            elif found_primitive:
                action_primitive_indices.append(var_i)
                action_primitives.append(found_primitive)

        # And now build the ultimate lookup table. Each entry
        # is a 3-tuple:
        # ( [node name, list of symbols that should be true],
        #   [possibly-empty list of primitives that can be taken] )
        self.state_lookup_table = {}
        for node_name in json_data["nodes"].keys():
            node_symbols = []
            node_state = json_data["nodes"][node_name]["state"]
            for i, sym in zip(environment_symbol_indices, environment_symbols):
                if node_state[i]:
                    node_symbols.append(sym)
            node_primitives = []
            for i, prim in zip(action_primitive_indices, action_primitives):
                if node_state[i]:
                    node_primitives.append(prim)
            self.state_lookup_table[node_name] = ((node_name, node_symbols, node_primitives))

        # TODO: Add state update method and plandata output.


class SymbolLoggerSystem(LeafSystem):
    ''' Consumes robot state and checks symbol status against it
    periodically, publishing to console. '''

    def __init__(self, mbp, grab_period=0.1, symbol_logger=None):
        LeafSystem.__init__(self)

        self.mbp = mbp
        # Our version of MBP context, which we'll modify
        # in the publish method.
        self.mbp_context = mbp.CreateDefaultContext()

        # Object body names we care about
        self.body_names = ["blue_box", "red_box"]

        self.set_name('symbol_detection_system')
        self.DeclarePeriodicPublish(grab_period, 0.0)

        # Take robot state vector as input.
        self.DeclareVectorInputPort("mbp_state_vector",
                                    BasicVector(mbp.num_positions() +
                                                mbp.num_velocities()))

        self._symbol_logger = symbol_logger

    def DoPublish(self, context, event):
        # TODO(russt): Change this to declare a periodic event with a
        # callback instead of overriding DoPublish, pending #9992.
        LeafSystem.DoPublish(self, context, event)

        mbp_state_vector = self.EvalVectorInput(context, 0).get_value()
        self.mbp.SetPositionsAndVelocities(self.mbp_context, mbp_state_vector)

        # Get pose of object
        for body_name in self.body_names:
            print(body_name, ": ")
            print(self.mbp.EvalBodyPoseInWorld(
                self.mbp_context, self.mbp.GetBodyByName(body_name)).matrix())

        rigid_transform_dict = {}
        for body_name in self.body_names:
            rigid_transform_dict[body_name] = self.mbp.EvalBodyPoseInWorld(self.mbp_context,
                                                                           self.mbp.GetBodyByName(body_name))
        self._symbol_logger.log_symbols(rigid_transform_dict)
        self._symbol_logger.print_curr_symbols()


def RegisterVisualAndCollisionGeometry(
        mbp, body, pose, shape, name, color, friction):
    ''' Register a Body subclass (usually RigidBody) to the MultibodyPlant
    with the specified shape, color, and friction. '''
    mbp.RegisterVisualGeometry(body, pose, shape, name + "_vis", color)
    mbp.RegisterCollisionGeometry(body, pose, shape, name + "_col",
                                  friction)

def add_box_at_location(mbp, name, color, pose, mass=0.1, inertia=UnitInertia(0.001, 0.001, 0.001)):
    ''' Adds a 5cm cube at the specified pose. Uses a planar floating base
    in the x-z plane. '''
    no_mass_no_inertia = SpatialInertia(
            mass=0., p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(0., 0., 0.))
    body_mass_and_inertia = SpatialInertia(
            mass=mass, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=inertia)
    shape = Box(0.05, 0.05, 0.05)
    model_instance = mbp.AddModelInstance(name)
    body = mbp.AddRigidBody(name, model_instance, body_mass_and_inertia)
    RegisterVisualAndCollisionGeometry(
        mbp, body,
        RigidTransform(),
        shape, name, color,
        CoulombFriction(0.9, 0.8))
    body_pre_z = mbp.AddRigidBody("{}_pre_z".format(name), model_instance,
                                  no_mass_no_inertia)
    body_pre_theta = mbp.AddRigidBody("{}_pre_theta".format(name), model_instance,
                                      no_mass_no_inertia)

    world_carrot_origin = mbp.AddFrame(frame=FixedOffsetFrame(
            name="world_{}_origin".format(name), P=mbp.world_frame(),
            X_PF=pose))
    body_joint_x = PrismaticJoint(
        name="{}_x".format(name),
        frame_on_parent=world_carrot_origin,
        frame_on_child=body_pre_z.body_frame(),
        axis=[1, 0, 0],
        damping=0.)
    mbp.AddJoint(body_joint_x)

    body_joint_z = PrismaticJoint(
        name="{}_z".format(name),
        frame_on_parent=body_pre_z.body_frame(),
        frame_on_child=body_pre_theta.body_frame(),
        axis=[0, 0, 1],
        damping=0.)
    mbp.AddJoint(body_joint_z)

    body_joint_theta = RevoluteJoint(
        name="{}_theta".format(name),
        frame_on_parent=body_pre_theta.body_frame(),
        frame_on_child=body.body_frame(),
        axis=[0, 1, 0],
        damping=0.)
    mbp.AddJoint(body_joint_theta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    MeshcatVisualizer.add_argparse_argument(parser)
    parser.add_argument('--use_meshcat', action='store_true',
                        help="Must be set for meshcat to be used.")
    parser.add_argument('--disable_planar_viz', action='store_true',
                        help="Don't create a planar visualizer. Probably"
                             " breaks something that assumes the planar"
                             " vis exists.")
    parser.add_argument('--teleop', action='store_true',
                        help="Enable teleop, so *don't* use the state machine"
                             " and motion primitives.")

    args = parser.parse_args()

    builder = DiagramBuilder()

    # Set up the ManipulationStation
    station = builder.AddSystem(ManipulationStation())
    mbp = station.get_multibody_plant()
    station.SetupManipulationClassStation()
    add_box_at_location(mbp, name="blue_box", color=[0.25, 0.25, 1., 1.],
                        pose=RigidTransform(p=[0.4, 0.0, 0.05]))
    add_box_at_location(mbp, name="red_box", color=[1., 0.25, 0.25, 1.],
                        pose=RigidTransform(p=[0.55, 0.0, 0.05]))
    station.Finalize()
    iiwa_q0 = np.array([0.0, 0.6, 0.0, -1.75, 0., 1., np.pi / 2.])

    # Attach a visualizer.
    if args.use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(
            station.get_scene_graph(), zmq_url=args.meshcat))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))
    
    if not args.disable_planar_viz:
        plt.gca().clear()
        viz = builder.AddSystem(PlanarSceneGraphVisualizer(
            station.get_scene_graph(),
            xlim=[0.25, 0.8], ylim=[-0.1, 0.5],
            ax=plt.gca()))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        viz.get_input_port(0))
        plt.draw()

    if not args.teleop:
        # Hook up DifferentialIK, since teleop will control
        # in end effector frame.
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
        params.set_joint_velocity_limits((-factor * iiwa14_velocity_limits,
                                          factor * iiwa14_velocity_limits))
        differential_ik = builder.AddSystem(DifferentialIK(
            robot, robot.GetFrameByName("iiwa_link_7"), params, time_step))
        differential_ik.parameters.set_nominal_joint_position(iiwa_q0)
        builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                        station.GetInputPort("iiwa_position"))


        goal_position = [0.5, 0., 0.05]
        goal_delta = 0.05
        symbol_list = [
            SymbolL2Close("blue_box_in_goal", "blue_box", goal_position, goal_delta),
            SymbolL2Close("red_box_in_goal", "red_box", goal_position, goal_delta),
        ]
        primitive_list = [
            IiwaAndGripperPrimitive("put_blue_box_in_goal", mbp),
            IiwaAndGripperPrimitive("put_red_box_in_goal", mbp),
            IiwaAndGripperPrimitive("clean_blue_box_from_goal", mbp),
            IiwaAndGripperPrimitive("clean_red_box_from_goal", mbp)
        ]
        task_execution_system = builder.AddSystem(
            TaskExectionSystem(
                mbp, symbol_list=symbol_list, primitive_list=primitive_list,
                dfa_json_file="red_and_blue_boxes.json"))

        movebox = MoveBoxPrimitive("test_move_box", mbp, "red_box", goal_position)
        rpy_xyz_trajectory, gripper_traj = movebox.generate_rpyxyz_and_gripper_trajectory(mbp.CreateDefaultContext())

        rpy_xyz_trajectory_source = builder.AddSystem(TrajectorySource(rpy_xyz_trajectory))
        builder.Connect(rpy_xyz_trajectory_source.get_output_port(0),
                        differential_ik.GetInputPort("rpy_xyz_desired"))

        # Target zero feedforward residual torque at all times.
        fft = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
        builder.Connect(fft.get_output_port(0),
                        station.GetInputPort("iiwa_feedforward_torque"))

        input_force_fix = builder.AddSystem(ConstantVectorSource([40.0]))
        builder.Connect(input_force_fix.get_output_port(0),
                        station.GetInputPort("wsg_force_limit"))
        wsg_position_source = builder.AddSystem(TrajectorySource(gripper_traj))
        builder.Connect(wsg_position_source.get_output_port(0),
                        station.GetInputPort("wsg_position"))

        end_time = rpy_xyz_trajectory.end_time() + 100.0

    else:  # Set up teleoperation.
        # Hook up DifferentialIK, since teleop will control
        # in end effector frame.
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
        params.set_joint_velocity_limits((-factor * iiwa14_velocity_limits,
                                          factor * iiwa14_velocity_limits))
        differential_ik = builder.AddSystem(DifferentialIK(
            robot, robot.GetFrameByName("iiwa_link_7"), params, time_step))
        differential_ik.parameters.set_nominal_joint_position(iiwa_q0)
        builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                        station.GetInputPort("iiwa_position"))

        # Hook up a pygame-based keyboard+mouse interface for
        # teleoperation, and low pass its output to drive the EE target
        # for the differential IK.
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

        # Target zero feedforward residual torque at all times.
        fft = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
        builder.Connect(fft.get_output_port(0),
                        station.GetInputPort("iiwa_feedforward_torque"))
        # Simulate functionally forever.
        end_time = 10000

    # Create symbol log
    symbol_log = SymbolFromTransformLog(
        [SymbolL2Close('at_goal', 'red_box', goal_position, .025),
         SymbolL2Close('at_goal', 'blue_box', goal_position, .025)])

    symbol_logger_system = builder.AddSystem(
        SymbolLoggerSystem(
            station.get_multibody_plant(), symbol_logger=symbol_log))
    builder.Connect(
        station.GetOutputPort("plant_continuous_state"),
        symbol_logger_system.GetInputPort("mbp_state_vector"))

    # Remaining input ports need to be tied up.
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    station_context = diagram.GetMutableSubsystemContext(
        station, diagram_context)

    station.SetIiwaPosition(station_context, iiwa_q0)
    differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
        differential_ik, diagram_context), iiwa_q0)
    
    if args.teleop:
        teleop.SetPose(differential_ik.ForwardKinematics(iiwa_q0))
        filter_.set_initial_output_value(
            diagram.GetMutableSubsystemContext(
                filter_, diagram_context),
            teleop.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(
                teleop, diagram_context)))

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(end_time)


if __name__ == "__main__":
    main()
