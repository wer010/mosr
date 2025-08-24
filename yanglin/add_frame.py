# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np
import threading
import time
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
# data = np.load("01_01_poses.npz")

# 导入smplh模型
smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
# 截取amass部分数据
# poses = data["poses"][0:2]
poses = np.zeros([1,165])
trans = np.zeros([1,3])
i_root_end = 3
i_body_end = i_root_end + smpl_layer.bm.NUM_BODY_JOINTS * 3
i_left_hand_end = i_body_end + 15 * 3
i_right_hand_end = i_left_hand_end + 15 * 3

pose_body_data = poses[:, i_root_end:i_body_end]
seq_smpl = SMPLSequence(
            smpl_layer=smpl_layer,
            poses_body=poses[:, i_root_end:i_body_end],
            poses_root=poses[:, :i_root_end],
            poses_left_hand=poses[:, i_body_end:i_left_hand_end],
            poses_right_hand=poses[:, i_left_hand_end:i_right_hand_end],
            poses_head = poses[:, i_right_hand_end:] if poses.shape[1]>i_right_hand_end else None,
            trans=trans,
            z_up=False
        )
def add_sequence_later(seq_smpl):
    t = 0
    while True:
        print("Adding T-pose...")
        t+=1
        # add_frames(self, poses_body, poses_root=None, trans=None, betas=None)
        poses = np.zeros([1, 165])
        a = np.array([1,0,1])
        # poses[0, 17*3:18*3] = ((t%20)/20)*(np.pi)*a/np.linalg.norm(a)
        trans = np.zeros([1, 3])  + np.array([[0, .5, 0]])*np.sin(t)

        seq_smpl.update_frames(poses_body=poses[0, i_root_end:i_body_end], frames = 0, trans=trans[0])

        time.sleep(0.1)  # 等1秒


v = Viewer()
# Timer(0.1,action).start()
# v.run_animations = True
# v.scene.camera.position = np.array([1.0, 2.5, 0.0])
v.scene.add(seq_smpl)
poses = np.zeros([1, 165])
trans = np.zeros([1, 3])+np.array([[0,0,0.5]])

v.scene.fps = 10
# v.scene.playback_fps = 1000
t = 0
threading.Thread(target=add_sequence_later, args=(seq_smpl,), daemon=True).start()



v.run()