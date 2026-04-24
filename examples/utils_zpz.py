import numpy as np
from scipy.spatial.transform import Rotation

class CameraPoseInterpolator:
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def compute_pose_distance(self, pose1, pose2):
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)
        R1 = Rotation.from_matrix(pose1[:3, :3])
        R2 = Rotation.from_matrix(pose2[:3, :3])
        q1, q2 = R1.as_quat(), R2.as_quat()
        if np.dot(q1, q2) < 0: q2 = -q2
        dot_val = np.clip(2 * np.dot(q1, q2)**2 - 1, -1.0, 1.0)
        rotation_dist = np.arccos(dot_val)
        return (self.translation_weight * translation_dist + self.rotation_weight * rotation_dist)

    def find_nearest_assignments(self, training_poses, testing_poses):
        assignments = []
        for j in range(len(testing_poses)):
            distances = [self.compute_pose_distance(tp, testing_poses[j]) for tp in training_poses]
            assignments.append(np.argmin(distances))
        return assignments

    def interpolate_rotation(self, R1, R2, t):
        q1 = Rotation.from_matrix(R1).as_quat()
        q2 = Rotation.from_matrix(R2).as_quat()
        if np.dot(q1, q2) < 0: q2 = -q2
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
        theta = np.arccos(dot_product)
        if np.abs(theta) < 1e-6:
            q_interp = (1 - t) * q1 + t * q2
        else:
            q_interp = (np.sin((1-t)*theta) * q1 + np.sin(t*theta) * q2) / np.sin(theta)
        q_interp = q_interp / np.linalg.norm(q_interp)
        return Rotation.from_quat(q_interp).as_matrix()

# ================================================================
    # 核心创新点二：流形锚定的局部微步探索 (MACE / LTMS)
    # ================================================================
    def shift_poses(self, 
                    training_poses, 
                    testing_poses, 
                    distance=0.03,        # 极其微小的后退距离
                    elevation_deg=2.0,    # 极其微小的仰角变化 (度)
                    **kwargs): 
        """
        基于局部切线空间的微步生成策略 (Local-Tangent Micro-Stepping)。
        完全摒弃全局中心，沿着相机自身的局部坐标轴进行平移和俯仰旋转，
        确保视锥体始终稳定锁定在已知的流形表面上，为扩散模型提供强有力的局部先验约束。
        """
        import scipy.spatial.transform as transform
        import numpy as np

        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        novel_poses = []

        # 将角度转换为弧度
        elevation_rad = np.deg2rad(elevation_deg)

        for test_idx, train_idx in enumerate(assignments):
            # 提取锚点相机的真实位姿
            train_pose = training_poses[train_idx]
            R_orig = train_pose[:3, :3]
            t_orig = train_pose[:3, 3]
            
            # 在 COLMAP/OpenCV 相机坐标系中：
            # X轴: 向右 (R_orig[:, 0])
            # Y轴: 向下 (R_orig[:, 1])
            # Z轴: 向前/镜头方向 (R_orig[:, 2])
            
            # ----------------------------------------------------
            # 步骤 1: 局部俯仰旋转 (Pitch)
            # 我们让相机围绕自己的 X 轴(局部右方向)进行微小的俯仰旋转
            # 这样可以在不改变目标物体在画面中心的情况下，看到更多车顶
            # ----------------------------------------------------
            # 构建一个只绕 X 轴旋转的局部旋转矩阵
            pitch_matrix = transform.Rotation.from_euler('x', -elevation_rad).as_matrix()
            # 将局部旋转叠加到全局旋转上 (矩阵右乘)
            R_new = R_orig @ pitch_matrix
            
            # ----------------------------------------------------
            # 步骤 2: 沿新视轴微量后退 (Translation along new Z)
            # 旋转后，我们沿着新的镜头方向往后退一点，确保画面边缘留出修补空间
            # ----------------------------------------------------
            cam_forward_new = R_new[:, 2] # 新的镜头前方向
            cam_up_new = -R_new[:, 1]     # 新的相机上方
            
            # 后退一点点 (沿着 -Z)
            t_new = t_orig - distance * cam_forward_new
            # 为了防止后退导致底部穿帮，可以配合极其微小的高度补偿
            t_new = t_new + (distance * 0.2) * cam_up_new

            # ----------------------------------------------------
            # 步骤 3: 合成最终的新视角流形位姿
            # ----------------------------------------------------
            new_pose = np.eye(4)
            new_pose[:3, :3] = R_new
            new_pose[:3, 3] = t_new
            
            novel_poses.append(new_pose)

        return np.array(novel_poses)