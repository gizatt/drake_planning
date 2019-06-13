import os
import sys

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.trajectories import PiecewisePolynomial


# TODO(gizatt) Primitives might want to subclass subclass,
# since whether they're valid is a symbol itself.
class IiwaAndGripperPrimitive(object):
    ''' Interface for primitive actions that can be run
    on the ManipulationStation (Kuka IIWA + Schunk gripper).
    Makes all decisions based on an MBP context.'''
    def __init__(self, name, mbp):
        self.name = name
        self.mbp = mbp

    def is_valid(self, mbp_context):
        raise NotImplementedError()

    def generate_rpyxyz_and_gripper_trajectory(self, mbp_context):
        raise NotImplementedError()


class MoveBoxPrimitive(IiwaAndGripperPrimitive):
    ''' Interface for primitive actions that can be run
    on the ManipulationStation (Kuka IIWA + Schunk gripper).
    Makes all decisions based on an MBP context.'''
    def __init__(self, name, mbp, body_name, goal_position):
        IiwaAndGripperPrimitive.__init__(self, name, mbp)
        self.goal_position = np.array(goal_position)
        self.goal_body = mbp.GetBodyByName(body_name)

    def is_valid(self, mbp_context):
        # TODO(gizatt) This is probably not always true
        return True

    def generate_rpyxyz_and_gripper_trajectory(self, mbp_context):
        start_position = self.mbp.EvalBodyPoseInWorld(mbp_context, self.goal_body).translation()
        start_ee_pose = self.mbp.EvalBodyPoseInWorld(
            mbp_context, self.mbp.GetBodyByName("iiwa_link_7"))
        grasp_offset = np.array([0.0, 0., 0.2]) # Amount *all* target points are shifted up
        up_offset = np.array([0., 0., 0.1])      # Additional amount to lift objects off of table
        # Timing:
        #    0       :  t_reach: move over object
        #    t_reach :  t_touch: move down to object
        #    t_touch :  t_grasp: close gripper
        #    t_grasp :  t_lift : pick up object
        #    t_lift  :  t_move : move object over destination
        #    t_move  :  t_down : move object down
        #    t_down  :  t_drop : open gripper
        #    t_drop  :  t_done : rise back up
        t_each = 0.25
        t_reach = 0. + t_each 
        t_touch = t_reach + t_each 
        t_grasp = t_touch + t_each 
        t_lift = t_grasp + t_each 
        t_move = t_lift + t_each 
        t_down = t_move + t_each 
        t_drop = t_down + t_each 
        t_done = t_drop + t_each 
        ts = np.array([0., t_reach, t_touch, t_grasp, t_lift, t_move, t_down, t_drop, t_done])
        ee_xyz_knots = np.vstack([
            start_ee_pose.translation() - grasp_offset,
            start_position + up_offset,
            start_position,
            start_position,
            start_position + up_offset,
            self.goal_position + up_offset,
            self.goal_position,
            self.goal_position,
            self.goal_position + up_offset,
        ]).T
        ee_xyz_knots += np.tile(grasp_offset, [len(ts), 1]).T

        ee_rpy_knots = np.array([-np.pi, 0., np.pi/2.])
        ee_rpy_knots = np.tile(ee_rpy_knots, [len(ts), 1]).T
        ee_rpyxyz_knots = np.vstack([ee_rpy_knots, ee_xyz_knots])
        ee_rpyxyz_traj = PiecewisePolynomial.FirstOrderHold(
            ts, ee_rpyxyz_knots)

        gripper_knots = np.array([[0.1, 0.1, 0.1, 0., 0., 0., 0., 0.1, 0.1]])
        gripper_traj = PiecewisePolynomial.FirstOrderHold(
            ts, gripper_knots)
        return ee_rpyxyz_traj, gripper_traj