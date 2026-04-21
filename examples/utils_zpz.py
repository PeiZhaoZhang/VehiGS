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
    # 核心可调接口
    # ================================================================
    def shift_poses(self, 
                    training_poses, 
                    testing_poses, 
                    distance=0.0, 
                    elevation_amount=0.4, 
                    center_offset=None,
                    up_axis_manual=None):
        """
        Args:
            training_poses: 原始训练位姿
            testing_poses: 原始测试位姿
            distance: [调节远近] 正数远离车，负数靠近车 (建议范围: -0.5 ~ 1.0)
            elevation_amount: [调节高度] 0.0是在地面，1.0是正上方 (建议范围: 0.3 ~ 0.6)
            center_offset: [中心微调] 如果发现车不在画面中心，输入 [x, y, z] 进行偏移
            up_axis_manual: [强制坐标轴] 如果车还是歪的，可以手动指定如 [0, 0, 1]
        """
        
        # 1. 确定车辆中心 (Look-at Target)
        center = np.mean(training_poses[:, :3, 3], axis=0)
        if center_offset is not None:
            center += np.array(center_offset)
            
        # 2. 自动提取“世界向上”向量 (自适应不同数据集)
        if up_axis_manual is not None:
            avg_up = np.array(up_axis_manual)
        else:
            # 假设 Y 轴平均指向下方，反方向为上 (适配手持数据)
            avg_up = -np.mean(training_poses[:, :3, 1], axis=0) 
            
        avg_up /= (np.linalg.norm(avg_up) + 1e-8)
        
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        novel_poses = []

        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            curr_pos = train_pose[:3, 3]
            
            # 3. 球面投影逻辑
            vec = curr_pos - center
            dist = np.linalg.norm(vec)
            radial_dir = vec / (dist + 1e-8)
            
            # 抬升：在“径向”和“向上”之间插值
            new_dir = (1 - elevation_amount) * radial_dir + elevation_amount * avg_up
            new_dir /= np.linalg.norm(new_dir)
            
            # 计算新位置
            new_t = center + new_dir * (dist + distance)

            # 4. 构建 Look-at 矩阵 (决定相机怎么看车)
            f = center - new_t  # 前方向 (Z)
            f /= (np.linalg.norm(f) + 1e-8)
            
            r = np.cross(f, avg_up)  # 右方向 (X)
            if np.linalg.norm(r) < 1e-6:
                r = np.array([1.0, 0.0, 0.0])
            r /= np.linalg.norm(r)
            
            u = np.cross(r, f)  # 上方向 (Y)
            
            # 组合成旋转矩阵 (适配 OpenCV 风格渲染器)
            # 如果画面依然是倒的，把 -u 改成 u
            R_new = np.stack([r, -u, f], axis=1) 

            new_pose = np.eye(4)
            new_pose[:3, :3] = R_new
            new_pose[:3, 3] = new_t
            novel_poses.append(new_pose)

        return np.array(novel_poses)