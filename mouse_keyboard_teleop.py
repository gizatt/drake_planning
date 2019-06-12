import pydrake

import argparse
import os
import random
import time
import sys

import numpy as np

import pygame
from pygame.locals import *

from pydrake.systems.framework import (BasicVector, LeafSystem)
from pydrake.math import RigidTransform, RollPitchYaw


def print_instructions():
    print("")
    print("END EFFECTOR CONTROL")
    print("mouse left/right   - move in the manipulation station's y/z plane")
    print("mouse buttons      - roll left/right")
    print("w / s              - move forward/back this y/z plane")
    print("q / e              - yaw left/right \
                                (also can use mouse side buttons)")
    print("a / d              - pitch up/down")
    print("")
    print("GRIPPER CONTROL")
    print("mouse wheel        - open/close gripper")
    print("")
    print("space              - switch out of teleop mode")
    print("enter              - return to teleop mode (be sure you've")
    print("                     returned focus to the pygame app)")
    print("escape             - quit")

class TeleopMouseKeyboardManager():

    def __init__(self, grab_focus=True):
        pygame.init()
        # We don't actually want a screen, but
        # I can't get this to work without a tiny screen.
        # Setting it to 1 pixel.
        screen_size = 1
        self.screen = pygame.display.set_mode((screen_size, screen_size))

        self.side_button_back_DOWN = False
        self.side_button_fwd_DOWN = False
        if grab_focus:
            self.grab_mouse_focus()

    def grab_mouse_focus(self):
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def release_mouse_focus(self):
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

    def get_events(self):
        mouse_wheel_up = mouse_wheel_down = False

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    mouse_wheel_up = True
                if event.button == 5:
                    mouse_wheel_down = True
                if event.button == 8:
                    self.side_button_back_DOWN = True
                if event.button == 9:
                    self.side_button_fwd_DOWN = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 8:
                    self.side_button_back_DOWN = False
                if event.button == 9:
                    self.side_button_fwd_DOWN = False

        keys = pygame.key.get_pressed()
        delta_x, delta_y = pygame.mouse.get_rel()
        left_mouse_button, _, right_mouse_button = pygame.mouse.get_pressed()

        if keys[K_RETURN]:
            self.grab_mouse_focus()
        if keys[K_SPACE]:
            self.release_mouse_focus()

        events = dict()
        events["delta_x"] = delta_x
        events["delta_y"] = delta_y
        events["w"] = keys[K_w]
        events["a"] = keys[K_a]
        events["s"] = keys[K_s]
        events["d"] = keys[K_d]
        events["r"] = keys[K_r]
        events["q"] = keys[K_q]
        events["e"] = keys[K_e]
        events["p"] = keys[K_p]
        events["mouse_wheel_up"] = mouse_wheel_up
        events["mouse_wheel_down"] = mouse_wheel_down
        events["left_mouse_button"] = left_mouse_button
        events["right_mouse_button"] = right_mouse_button
        events["side_button_back"] = self.side_button_back_DOWN
        events["side_button_forward"] = self.side_button_fwd_DOWN
        return events


class MouseKeyboardTeleop(LeafSystem):
    def __init__(self, grab_focus=True):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("rpy_xyz", BasicVector(6),
                                     self.DoCalcOutput)
        self.DeclareVectorOutputPort("position", BasicVector(1),
                                     self.CalcPositionOutput)
        self.DeclareVectorOutputPort("force_limit", BasicVector(1),
                                     self.CalcForceLimitOutput)

        # Note: This timing affects the keyboard teleop performance. A larger
        #       time step causes more lag in the response.
        self.DeclarePeriodicPublish(0.01, 0.0)

        self.teleop_manager = TeleopMouseKeyboardManager(grab_focus=grab_focus)
        self.roll = self.pitch = self.yaw = 0
        self.x = self.y = self.z = 0
        self.gripper_max = 0.107
        self.gripper_min = 0.01
        self.gripper_goal = self.gripper_max
        self.p_down = False

    def SetPose(self, pose):
        """
        @param pose is an Isometry3.
        """
        tf = RigidTransform(pose)
        self.SetRPY(RollPitchYaw(tf.rotation()))
        self.SetXYZ(pose.translation())

    def SetRPY(self, rpy):
        """
        @param rpy is a RollPitchYaw object
        """
        self.roll = rpy.roll_angle()
        self.pitch = rpy.pitch_angle()
        self.yaw = rpy.yaw_angle()

    def SetXYZ(self, xyz):
        """
        @param xyz is a 3 element vector of x, y, z.
        """
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]

    def SetXyzFromEvents(self, events):
        scale_down = 0.0001
        delta_x = events["delta_x"]*-scale_down
        delta_y = events["delta_y"]*-scale_down

        forward_scale = 0.00005
        delta_forward = 0.0
        if events["w"]:
            delta_forward += forward_scale
        if events["s"]:
            delta_forward -= forward_scale

        self.x += -delta_x
        self.y += delta_forward
        self.z += delta_y

    def SetRpyFromEvents(self, events):
        roll_scale = 0.0003
        if events["left_mouse_button"]:
            self.roll += roll_scale
        if events["right_mouse_button"]:
            self.roll -= roll_scale
        self.roll = np.clip(self.roll, a_min=-2*np.pi, a_max=2*np.pi)

        yaw_scale = 0.0003
        if events["side_button_back"] or events["q"]:
            self.yaw += yaw_scale
        if events["side_button_forward"] or events["e"]:
            self.yaw -= yaw_scale
        self.yaw = np.clip(self.yaw, a_min=-2*np.pi, a_max=2*np.pi)

        pitch_scale = 0.0003
        if events["d"]:
            self.pitch += pitch_scale
        if events["a"]:
            self.pitch -= pitch_scale
        self.pitch = np.clip(self.pitch, a_min=-2*np.pi, a_max=2*np.pi)

    def SetGripperFromEvents(self, events):
        gripper_scale = 0.01
        if events["mouse_wheel_up"]:
            self.gripper_goal += gripper_scale
        if events["mouse_wheel_down"]:
            self.gripper_goal -= gripper_scale
        self.gripper_goal = np.clip(self.gripper_goal,
                                    a_min=self.gripper_min,
                                    a_max=self.gripper_max)

    def CalcPositionOutput(self, context, output):
        output.SetAtIndex(0, self.gripper_goal)

    def CalcForceLimitOutput(self, context, output):
        self._force_limit = 40
        output.SetAtIndex(0, self._force_limit)

    def DoCalcOutput(self, context, output):
        events = self.teleop_manager.get_events()
        self.SetXyzFromEvents(events)
        self.SetRpyFromEvents(events)
        self.SetGripperFromEvents(events)
        output.SetAtIndex(0, self.roll)
        output.SetAtIndex(1, self.pitch)
        output.SetAtIndex(2, self.yaw)
        output.SetAtIndex(3, self.x)
        output.SetAtIndex(4, self.y)
        output.SetAtIndex(5, self.z)
        if (not self.p_down and events["p"]):
            print("Pose: ", output.CopyToVector())
            self.p_down = True
        elif (not events["p"]):
            self.p_down = False