import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface,
    CreateDefaultYcbObjectList)
from pydrake.geometry import (
    Box, Sphere
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
    AbstractValue, BasicVector, DiagramBuilder, LeafSystem, 
    UnrestrictedUpdateEvent)
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

    class TESState(object):
        start_time = 0.
        current_state_name = "0"
        rpy_xyz_desired_traj = None
        wsg_position_traj = None

    def __init__(self, mbp, symbol_list, primitive_list, dfa_json_file, update_period=0.1):
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

        # State of system is current RPYXYZ and gripper goals
        self.DeclareAbstractState(AbstractValue.Make(self.TESState()))
        self.DeclarePeriodicEvent(
            period_sec=update_period, offset_sec=0.0,
            event=UnrestrictedUpdateEvent(self.UpdateAbstractState))


        # Produce RPYXYZ goals + gripper goals
        self.DeclareVectorOutputPort("rpy_xyz_desired",
                                     BasicVector(6), self.CopyRpyXyzDesiredOut)
        self.DeclareVectorOutputPort("wsg_position",
                                     BasicVector(1), self.CopyWsgPositionOut)

        # Load the JSON file
        with open(dfa_json_file, 'r') as f:
            json_data = json.load(f)

        # Figure out which states in the JSON dict are environment
        # symbols, and which are primitives that we can execute.
        self.environment_symbol_indices = []
        self.environment_symbols = []
        self.action_primitive_indices = []
        self.action_primitives = []

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
                self.environment_symbol_indices.append(var_i)
                self.environment_symbols.append(found_symbol)
            elif found_primitive:
                self.action_primitive_indices.append(var_i)
                self.action_primitives.append(found_primitive)

        # And now build the ultimate lookup dictionary containing,
        # at each node name:
        # ( [environment symbol truth vector for this state],
        #   [possibly-empty list of primitives that can be taken,
        #   [next state names]] )
        self.state_lookup_table = {}
        for node_name in json_data["nodes"].keys():
            node_symbols = []
            node_state = np.array(json_data["nodes"][node_name]["state"])
            next_node_names = [str(x) for x in json_data["nodes"][node_name]["trans"]]
            node_primitives = []
            for i, prim in zip(self.action_primitive_indices, self.action_primitives):
                if node_state[i]:
                    node_primitives.append(prim)
            self.state_lookup_table[node_name] = (
                 node_state[self.environment_symbol_indices],
                 node_primitives, next_node_names
            )

    def UpdateAbstractState(self, context, event, state):
        x = self.EvalVectorInput(context, 0).get_value()
        t = context.get_time()
        self.mbp.SetPositionsAndVelocities(self.mbp_context, x)
        state_object = context.get_state().get_abstract_state().get_value(0).get_value()

        trajectory_is_done = (
            state_object.rpy_xyz_desired_traj is None or
            state_object.wsg_position_traj is None or 
            (t - state_object.start_time >= state_object.rpy_xyz_desired_traj.end_time() and
            t - state_object.start_time >= state_object.wsg_position_traj.end_time())
        )
        if trajectory_is_done:
            # Check each symbol against current mbp_context
            curr_state_vector = []
            for symbol in self.environment_symbols:
                # TODO(gizatt) Generalize symbol interface to accept mbp context and
                # do this lookup itself.
                pose_dict = {}
                for name in symbol._object_names:
                    pose_dict[name] = self.mbp.EvalBodyPoseInWorld(self.mbp_context, self.mbp.GetBodyByName(name))
                curr_state_vector.append(symbol(pose_dict))
            curr_state_vector = np.array(curr_state_vector)
            print("Current state vector: ", curr_state_vector)

            # Repeatedly step state forward, following desired transitions,
            # until we find a state with an executable primitive.
            primitive = None
            while primitive is None:
                # Check which states we're allowed to transition to.
                matching_states = []
                for node_name in self.state_lookup_table.keys():
                    if np.all(self.state_lookup_table[node_name][0] == curr_state_vector):
                        matching_states.append(node_name)
                if len(matching_states) == 0:
                    raise ValueError("No matching states to current symbol set.")

                # Check which states we *want* to transition to.
                good_transitions = []
                if state_object.current_state_name is not None:
                    for next_node_name in self.state_lookup_table[state_object.current_state_name][2]:
                        if next_node_name in matching_states:
                            good_transitions.append(next_node_name)

                if len(good_transitions) == 0:
                    print("WARNING: no good transitions from last state. Relocalizing to random valid state...")
                    good_transitions.append(random.choice(matching_states))

                # Pick random good transition and follow it.
                new_state_name = random.choice(good_transitions)
                print("Transitioning from {} to {}".format(state_object.current_state_name, new_state_name))
                state_object.current_state_name = random.choice(good_transitions)
                
                primitive_options = self.state_lookup_table[state_object.current_state_name][1]
                if len(primitive_options) > 0:
                    primitive = random.choice(primitive_options)

            print("Applying primitive {}".format(primitive.name))
            state_object.rpy_xyz_desired_traj, state_object.wsg_position_traj = \
                primitive.generate_rpyxyz_and_gripper_trajectory(self.mbp_context)
            state_object.start_time = context.get_time()
            state.get_mutable_abstract_state().get_value(0).set_value(state_object)


    def CopyRpyXyzDesiredOut(self, context, output):
        t = context.get_time()
        # yuuuuck
        state_object = context.get_state().get_abstract_state().get_value(0).get_value()
        traj = state_object.rpy_xyz_desired_traj
        if traj is None:
            raise NotImplementedError("Traj should always be set before output is calculated.")
        output.SetFromVector(traj.value(t - state_object.start_time))

    def CopyWsgPositionOut(self, context, output):
        t = context.get_time()
        state_object = context.get_state().get_abstract_state().get_value(0).get_value()
        traj = state_object.wsg_position_traj
        if traj is None:
            raise NotImplementedError("Traj should always be set before output is calculated.")
        output.SetFromVector(traj.value(t - state_object.start_time))


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

def add_box_at_location(mbp, name, color, pose, mass=0.25, inertia=UnitInertia(3E-3, 3E-3, 3E-3)):
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

def add_goal_region_visual_geometry(mbp, goal_position, goal_delta):
    ''' Adds a 5cm cube at the specified pose. Uses a planar floating base
    in the x-z plane. '''
    shape = Box(goal_delta, goal_delta, goal_delta)
    no_mass_no_inertia = SpatialInertia(
            mass=0., p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(0., 0., 0.))
    shape = Sphere(0.05)
    model_instance = mbp.AddModelInstance("goal_vis")
    vis_origin_frame = mbp.AddFrame(frame=FixedOffsetFrame(
            name="goal_vis_origin", P=mbp.world_frame(),
            X_PF=RigidTransform(p=(goal_position + np.array([0., 0.5, 0.])))))
    body = mbp.AddRigidBody("goal_vis", model_instance, no_mass_no_inertia)

    mbp.WeldFrames(vis_origin_frame, body.body_frame())
    mbp.RegisterVisualGeometry(body, RigidTransform(), shape, "goal_vis", [0.4, 0.9, 0.4, 0.35])

def main():
    goal_position = np.array([0.5, 0., 0.025])
    blue_box_clean_position = [0.4, 0., 0.05]
    red_box_clean_position = [0.6, 0., 0.05]
    goal_delta = 0.05

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
    station = builder.AddSystem(ManipulationStation(0.001))
    mbp = station.get_multibody_plant()
    station.SetupManipulationClassStation()
    add_goal_region_visual_geometry(mbp, goal_position, goal_delta)
    add_box_at_location(mbp, name="blue_box", color=[0.25, 0.25, 1., 1.],
                        pose=RigidTransform(p=[0.4, 0.0, 0.025]))
    add_box_at_location(mbp, name="red_box", color=[1., 0.25, 0.25, 1.],
                        pose=RigidTransform(p=[0.6, 0.0, 0.025]))
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

    # Hook up DifferentialIK, since both control modes use it.
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

    if not args.teleop:
        symbol_list = [
            SymbolL2Close("blue_box_in_goal", "blue_box", goal_position, goal_delta),
            SymbolL2Close("red_box_in_goal", "red_box", goal_position, goal_delta),
            SymbolRelativePositionL2("blue_box_on_red_box", "blue_box", "red_box", l2_thresh=0.01, offset=np.array([0., 0., 0.05])),
            SymbolRelativePositionL2("red_box_on_blue_box", "red_box", "blue_box", l2_thresh=0.01, offset=np.array([0., 0., 0.05])),
        ]
        primitive_list = [
            MoveBoxPrimitive("put_blue_box_in_goal", mbp, "blue_box", goal_position),
            MoveBoxPrimitive("put_red_box_in_goal", mbp, "red_box", goal_position),
            MoveBoxPrimitive("put_blue_box_away", mbp, "blue_box", blue_box_clean_position),
            MoveBoxPrimitive("put_red_box_away", mbp, "red_box", red_box_clean_position),
            MoveBoxPrimitive("put_red_box_on_blue_box", mbp, "red_box", np.array([0., 0., 0.05]), "blue_box"),
            MoveBoxPrimitive("put_blue_box_on_red_box", mbp, "blue_box", np.array([0., 0., 0.05]), "red_box"),
        ]
        task_execution_system = builder.AddSystem(
            TaskExectionSystem(
                mbp, symbol_list=symbol_list, primitive_list=primitive_list,
                dfa_json_file="red_and_blue_boxes_stacking.json"))

        builder.Connect(
            station.GetOutputPort("plant_continuous_state"),
            task_execution_system.GetInputPort("mbp_state_vector"))
        builder.Connect(task_execution_system.get_output_port(0),
                        differential_ik.GetInputPort("rpy_xyz_desired"))
        builder.Connect(task_execution_system.get_output_port(1),
                        station.GetInputPort("wsg_position"))

        #movebox = MoveBoxPrimitive("test_move_box", mbp, "red_box", goal_position)
        #rpy_xyz_trajectory, gripper_traj = movebox.generate_rpyxyz_and_gripper_trajectory(mbp.CreateDefaultContext())
        #rpy_xyz_trajectory_source = builder.AddSystem(TrajectorySource(rpy_xyz_trajectory))
        #builder.Connect(rpy_xyz_trajectory_source.get_output_port(0),
        #                differential_ik.GetInputPort("rpy_xyz_desired"))
        #wsg_position_source = builder.AddSystem(TrajectorySource(gripper_traj))
        #builder.Connect(wsg_position_source.get_output_port(0),
        #                station.GetInputPort("wsg_position"))

        # Target zero feedforward residual torque at all times.
        fft = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
        builder.Connect(fft.get_output_port(0),
                        station.GetInputPort("iiwa_feedforward_torque"))

        input_force_fix = builder.AddSystem(ConstantVectorSource([40.0]))
        builder.Connect(input_force_fix.get_output_port(0),
                        station.GetInputPort("wsg_force_limit"))

        end_time = 10000

    else:  # Set up teleoperation.
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
    #symbol_log = SymbolFromTransformLog(
    #    [SymbolL2Close('at_goal', 'red_box', goal_position, .025),
    #     SymbolL2Close('at_goal', 'blue_box', goal_position, .025)])
#
    #symbol_logger_system = builder.AddSystem(
    #    SymbolLoggerSystem(
    #        station.get_multibody_plant(), symbol_logger=symbol_log))
    #builder.Connect(
    #    station.GetOutputPort("plant_continuous_state"),
    #    symbol_logger_system.GetInputPort("mbp_state_vector"))

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
