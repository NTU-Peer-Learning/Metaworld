import numpy as np
from gymnasium.spaces import Box
from metaworld.envs import reward_utils
from metaworld.envs.mujoco.arm_env import ArmEnv
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set


class JacoEnv(ArmEnv):
    _ACTION_DIM = 8
    _QPOS_SPACE = Box(
        np.array(
            [
                -3.14,
                0.82,
                -3.14,
                0.524,
                -3.14,
                1.13,
                -3.14,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        np.array(
            [
                3.14,
                5.46,
                3.14,
                5.76,
                3.14,
                5.15,
                3.14,
                1.51,
                2,
                1.51,
                2,
                1.51,
                2,
            ]
        ),
        dtype=np.float64,
    )

    def __init__(
        self,
        model_name,
        hand_low=...,
        hand_high=...,
        mocap_low=None,
        mocap_high=None,
        render_mode=None,
    ):
        super().__init__(
            model_name=model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            mocap_low=mocap_low,
            mocap_high=mocap_high,
            render_mode=render_mode,
        )

        self.hand_init_qpos = np.array(
            [0, 3.680, 0.000, 1.170, 0.050, 3.760, 0, 0.5, 0, 0.5, 0, 0.5, 0]
        )
        # TODO
        # self.init_finger_1 = self.get_body_com("jaco_link_finger_1")
        # self.init_finger_2 = self.get_body_com("jaco_link_finger_2")
        # self.init_finger_3 = self.get_body_com("jaco_link_finger_3")

        self.action_space = Box(
            np.array(
                [
                    -30.5,
                    -30.5,
                    -30.5,
                    -30.5,
                    -30.5,
                    0,
                    0,
                    0,
                ]
            ),
            np.array(
                [
                    30.5,
                    30.5,
                    30.5,
                    30.5,
                    30.5,
                    1.51,
                    1.51,
                    1.51,
                ]
            ),
            dtype=np.float64,
        )

        self.arm_col = [
            "ah1_collision",
            "ah2_collision",
            "f_collision",
            "ws1_collision",
            "ws2_collision",
            "right_l1_2",
            "right_l2_2",
            "right_l4_2",
        ]

        self.action_cost_coff = 1e-3

        self.init_left_pad = self.get_body_com("index_distal")
        self.init_right_pad = self.get_body_com("thumb_distal")

    @property
    def tcp_center(self):
        """The COM of the gripper's 3 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        finger_1, finger_2, finger_3 = (
            self.data.body("pinky_distal"),
            self.data.body("index_distal"),
            self.data.body("thumb_distal"),
        )
        tcp_center = (finger_1.xpos + finger_2.xpos + finger_3.xpos) / 3.0
        return tcp_center

    @property
    def left_pad(self):
        return self.get_body_com("index_distal")

    @property
    def right_pad(self):
        return self.get_body_com("thumb_distal")

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): 9-element array of actions
        """

        parsed_action = np.hstack((action, action[-1], action[-1]))
        self.do_simulation(parsed_action, n_frames=self.frame_skip)

    def gripper_effort_from_action(self, action):
        return action[-1] / 1.51

    def get_action_penalty(self, action):
        action_norm = np.linalg.norm(action)
        contact = self.check_contact_table()

        penalty = self.action_cost_coff * action_norm
        if contact:
            penalty = 5

        return penalty

    @property
    def gripper_distance_apart(self):
        # TODO
        return 0

    def touching_object(self, object_geom_id):
        # TODO
        return False

    def _gripper_caging_reward(  # TODO
        self,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
    ):
        # TODO
        return 0
