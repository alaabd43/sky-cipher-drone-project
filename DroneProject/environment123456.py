import pybullet as p
import time
import numpy as np
import os
import random
from collections import deque
import matplotlib.pyplot as plt

class DroneEnvironment:
    def __init__(self, render=False, level=1, enable_data_logging=False):
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        self.level = level
        self.radar_lines = []
        self.obstacle_ids = []
        self.tower_parts = []
        self.drone_crashed = False
        self.drone_id = None
        self.target_id = None
        self.ground_id = None
        self.beacon_id = None  

        self.enemy_patrols = []
        self.enemy_radar_bases = []
        self.enemy_radar_dishes = []
        self.all_enemy_ids = []
        self.radar_angles = []
        self.radar_positions = []
        self.radar_ranges = []

        self.wind_strength = 0.0
        self.wind_direction = np.array([0.0, 0.0, 0.0])
        self.wind_timer = 0
        self.wind_change_interval = 120
        self.wind_particles = deque()

        self.drone_detected_by_radar = False
        self._detection_this_step = False
        self.enable_data_logging = enable_data_logging
        self.current_episode_data = []
        self.episode_number = 0

        self.camera_width  = 64
        self.camera_height = 64
        self.camera_fov    = 85    
        self.camera_near   = 0.1
        self.camera_far    = 100.0
        self.camera_tilt_deg = 25  

    def reset(self):
        p.resetSimulation(self.client)
        self.drone_crashed = False
        self.enemy_patrols = []
        self.enemy_radar_bases = []
        self.enemy_radar_dishes = []
        self.all_enemy_ids = []
        self.tower_parts = []
        self.radar_angles = []
        self.radar_positions = []
        self.radar_ranges = []
        self.wind_particles.clear()
        self.wind_timer = 0
        self.radar_lines = []
        self.drone_detected_by_radar = False
        self._detection_this_step = False
        self.current_episode_data = []
        self.beacon_id = None
        self.episode_number += 1

        p.setGravity(0, 0, 0, physicsClientId=self.client)

        ground_shape  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1], physicsClientId=self.client)
        ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1], rgbaColor=[0.15, 0.5, 0.15, 1], physicsClientId=self.client)
        self.ground_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, -0.1],
            physicsClientId=self.client
        )

        if self.level >= 2:
            self.obstacle_ids = self.create_lego_city()
        else:
            self.obstacle_ids = []

        target_pos = self._get_safe_target_pos()
        target_pos[2] = 0.0
        self._create_relay_tower(target_pos)

        if self.level >= 3:
            self._reset_wind()
            self._spawn_enemy_patrols(num_enemies=3)
            self._spawn_ground_radars(num_radars=2, target_pos=target_pos)
            self._spawn_tanks(num_tanks=1)

        drone_path = os.path.join("drone_assets", "quadrotor.urdf")
        self.drone_id = p.loadURDF(drone_path, [0, 0, 3], globalScaling=2.0, physicsClientId=self.client)

    def _create_relay_tower(self, pos):
        x, y = pos[0], pos[1]
        base_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=1.8, length=0.5, rgbaColor=[0.45, 0.45, 0.45, 1.0], physicsClientId=self.client)
        base_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=1.8, height=0.5, physicsClientId=self.client)
        base_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_col, baseVisualShapeIndex=base_vis, basePosition=[x, y, 0.25], physicsClientId=self.client)
        self.tower_parts.append(base_id)

        target_col = p.createCollisionShape(p.GEOM_SPHERE, radius=2.0, physicsClientId=self.client)
        target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=2.0, rgbaColor=[1.0, 1.0, 0.0, 0.0], physicsClientId=self.client)
        self.target_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_col, baseVisualShapeIndex=target_vis, basePosition=[x, y, 3.5], physicsClientId=self.client)

        rim_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=1.85, length=0.12, rgbaColor=[1.0, 0.85, 0.0, 1.0], physicsClientId=self.client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=rim_vis, basePosition=[x, y, 0.44], physicsClientId=self.client)

        tower_low_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 2.5], rgbaColor=[0.30, 0.30, 0.30, 1.0], physicsClientId=self.client)
        tower_low_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 2.5], physicsClientId=self.client)
        tower_low_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=tower_low_col, baseVisualShapeIndex=tower_low_vis, basePosition=[x, y, 3.0], physicsClientId=self.client)
        self.tower_parts.append(tower_low_id)

        band_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.38, 0.38, 0.28], rgbaColor=[1.0, 0.40, 0.0, 1.0], physicsClientId=self.client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=band_vis, basePosition=[x, y, 5.5], physicsClientId=self.client)

        tower_top_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.28, 0.28, 1.8], rgbaColor=[0.25, 0.25, 0.25, 1.0], physicsClientId=self.client)
        tower_top_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.28, 0.28, 1.8], physicsClientId=self.client)
        tower_top_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=tower_top_col, baseVisualShapeIndex=tower_top_vis, basePosition=[x, y, 7.6], physicsClientId=self.client)
        self.tower_parts.append(tower_top_id)

        arm_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.08, 0.08], rgbaColor=[0.55, 0.55, 0.55, 1.0], physicsClientId=self.client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=arm_vis, basePosition=[x + 0.7, y, 7.2], physicsClientId=self.client)
        
        dish_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.55, rgbaColor=[0.80, 0.80, 0.80, 1.0], physicsClientId=self.client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=dish_vis, basePosition=[x + 1.3, y, 7.2], physicsClientId=self.client)

        antenna_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=2.2, rgbaColor=[0.70, 0.70, 0.70, 1.0], physicsClientId=self.client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=antenna_vis, basePosition=[x, y, 10.5], physicsClientId=self.client)

        warning_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.22, rgbaColor=[1.0, 0.10, 0.0, 1.0], physicsClientId=self.client)
        self.beacon_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=warning_vis, basePosition=[x, y, 11.7], physicsClientId=self.client)

    def _get_safe_target_pos(self):
        for _ in range(50):
            tx = random.uniform(-10, 10)
            ty = random.uniform(-10, 10)
            if abs(tx) < 3 and abs(ty) < 3:
                continue
            return [tx, ty, 0.0]
        return [6, 6, 0.0]

    def _reset_wind(self):
        angle = random.uniform(0, 2 * np.pi)
        self.wind_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        self.wind_strength  = random.uniform(0.5, 1.5)

    def _update_wind(self):
        self.wind_timer += 1
        if self.wind_timer >= self.wind_change_interval:
            self.wind_timer = 0
            angle_delta   = random.uniform(-np.pi / 3, np.pi / 3)
            current_angle = np.arctan2(self.wind_direction[1], self.wind_direction[0])
            new_angle     = current_angle + angle_delta
            self.wind_direction = np.array([np.cos(new_angle), np.sin(new_angle), 0.0])
            self.wind_strength  = np.clip(self.wind_strength + random.uniform(-0.4, 0.4), 0.3, 2.5)

    def get_wind_obs(self):
        if self.level < 3:
            return [0.0, 0.0, 0.0]
        return [
            float(self.wind_direction[0] * self.wind_strength),
            float(self.wind_direction[1] * self.wind_strength),
            float(self.wind_strength)
        ]

    def _spawn_enemy_patrols(self, num_enemies=3):
        drone_model_path = os.path.join("drone_assets", "quadrotor.urdf")
        for i in range(num_enemies):
            wp1 = [random.uniform(-12, 12), random.uniform(-12, 12), random.uniform(2, 6)]
            wp2 = [random.uniform(-12, 12), random.uniform(-12, 12), random.uniform(2, 6)]
            enemy_id  = p.loadURDF(drone_model_path, wp1, physicsClientId=self.client)
            num_joints = p.getNumJoints(enemy_id, physicsClientId=self.client)
            for j in range(-1, num_joints):
                p.changeVisualShape(enemy_id, j, rgbaColor=[0.8, 0.0, 0.0, 1], physicsClientId=self.client)
            p.changeDynamics(enemy_id, -1, linearDamping=1.0, angularDamping=1.0, physicsClientId=self.client)
            self.enemy_patrols.append({'id': enemy_id, 'waypoints': [wp1, wp2], 'current_wp': 1, 'speed': random.uniform(1.5, 3.0)})
            self.all_enemy_ids.append(enemy_id)

    def _update_enemy_behavior(self):
        if self.drone_id is None: return
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        drone_pos = np.array(drone_pos)

        for enemy in self.enemy_patrols:
            eid = enemy['id']
            pos, _ = p.getBasePositionAndOrientation(eid, physicsClientId=self.client)
            pos = np.array(pos)

            if self.drone_detected_by_radar:
                direction = drone_pos - pos
                dist = np.linalg.norm(direction)
                if dist > 0.2:
                    attack_speed = enemy['speed'] * 2.5 
                    vel = (direction / dist) * attack_speed
                    p.resetBaseVelocity(eid, linearVelocity=vel.tolist(), physicsClientId=self.client)
            else:
                direction = np.array(enemy['waypoints'][enemy['current_wp']]) - pos
                dist      = np.linalg.norm(direction)
                if dist < 0.5:
                    enemy['current_wp'] = 1 - enemy['current_wp']
                else:
                    vel = (direction / dist) * enemy['speed']
                    p.resetBaseVelocity(eid, linearVelocity=vel.tolist(), physicsClientId=self.client)

    def _spawn_ground_radars(self, num_radars=2, target_pos=None):
        if target_pos is None: target_pos = [0, 0, 0]
        for i in range(num_radars):
            angle = random.uniform(0, 2 * np.pi)
            dist  = random.uniform(3, 6)
            rx, ry = np.clip(target_pos[0] + dist * np.cos(angle), -13, 13), np.clip(target_pos[1] + dist * np.sin(angle), -13, 13)

            base_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.4, height=0.8, physicsClientId=self.client)
            base_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=0.8, rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.client)
            base_id  = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_col, baseVisualShapeIndex=base_vis, basePosition=[rx, ry, 0.4], physicsClientId=self.client)

            dish_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.1, 0.3], physicsClientId=self.client)
            dish_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.1, 0.3], rgbaColor=[0.7, 0.85, 1.0, 1], physicsClientId=self.client)
            dish_id  = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=dish_col, baseVisualShapeIndex=dish_vis, basePosition=[rx, ry, 1.1], physicsClientId=self.client)

            self.enemy_radar_bases.append(base_id)
            self.enemy_radar_dishes.append(dish_id)
            self.radar_angles.append(random.uniform(0, 2 * np.pi))
            self.radar_positions.append(np.array([rx, ry, 1.1]))
            self.radar_ranges.append(4.0)
            self.all_enemy_ids.extend([base_id, dish_id])

    def _spawn_tanks(self, num_tanks=1):
        for _ in range(num_tanks):
            tx, ty = random.uniform(-12, 12), random.uniform(-12, 12)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.8, 0.5, 0.4], rgbaColor=[0.3, 0.4, 0.1, 1], physicsClientId=self.client)
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.5, 0.4], physicsClientId=self.client)
            tank_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[tx, ty, 0.4], physicsClientId=self.client)
            self.all_enemy_ids.append(tank_id)

    def _update_ground_radars(self):
        if self.drone_id is None: return
        drone_pos = np.array(p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)[0])
        drone_detected = False

        for i, dish_id in enumerate(self.enemy_radar_dishes):
            self.radar_angles[i] += 0.03
            p.resetBasePositionAndOrientation(dish_id, p.getBasePositionAndOrientation(dish_id, physicsClientId=self.client)[0], p.getQuaternionFromEuler([0, 0, self.radar_angles[i]]), physicsClientId=self.client)
            if np.linalg.norm(drone_pos - self.radar_positions[i]) < self.radar_ranges[i]:
                drone_detected = True
                p.changeVisualShape(dish_id, -1, rgbaColor=[1.0, 0.0, 0.0, 1.0], physicsClientId=self.client)
            else:
                p.changeVisualShape(dish_id, -1, rgbaColor=[0.7, 0.85, 1.0, 1.0], physicsClientId=self.client)

        self.drone_detected_by_radar = drone_detected
        self._detection_this_step = drone_detected

    def create_lego_city(self, num_buildings=15, area_size=15):
        building_ids = []
        for _ in range(num_buildings):
            dims = [random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(1, 8)]
            pos  = [random.uniform(-area_size, area_size), random.uniform(-area_size, area_size), dims[2] / 2]
            if -3 < pos[0] < 3 and -3 < pos[1] < 3: continue
            gray_value = random.uniform(0.3, 0.7)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[d / 2 for d in dims], rgbaColor=[gray_value, gray_value, gray_value, 1], physicsClientId=self.client)
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d / 2 for d in dims], physicsClientId=self.client)
            building_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=pos, physicsClientId=self.client))
        return building_ids

    def _check_collisions(self):
        if self.drone_crashed: return
        all_dangers = self.obstacle_ids + self.all_enemy_ids + [self.ground_id] + self.tower_parts
        for danger_id in all_dangers:
            if p.getContactPoints(bodyA=self.drone_id, bodyB=danger_id, physicsClientId=self.client):
                self.drone_crashed = True
                return

    def update_radar(self):
        for line in self.radar_lines: p.removeUserDebugItem(line, physicsClientId=self.client)
        self.radar_lines.clear()
        if self.drone_crashed: return
        drone_pos = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)[0]

        for i in range(8):
            angle = 2 * np.pi * i / 8
            ray_to = [drone_pos[0] + 15 * np.cos(angle), drone_pos[1] + 15 * np.sin(angle), drone_pos[2]]
            ray_test = p.rayTest(drone_pos, ray_to, physicsClientId=self.client)
            hit_id, hit_pos = ray_test[0][0], ray_test[0][3]

            if hit_id != -1:
                color = [1, 1, 0] if hit_id == self.target_id else [0.6, 0.0, 1.0] if hit_id in self.all_enemy_ids else [1, 0, 0]
                self.radar_lines.append(p.addUserDebugLine(drone_pos, hit_pos, color, 2, 0.1, physicsClientId=self.client))

    def get_radar_obs(self):
        if self.drone_crashed or self.drone_id is None: return [1.0] * 8
        drone_pos = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)[0]
        radar_dists = []

        for i in range(8):
            angle = 2 * np.pi * i / 8
            ray_to = [drone_pos[0] + 15 * np.cos(angle), drone_pos[1] + 15 * np.sin(angle), drone_pos[2]]
            ray_test = p.rayTest(drone_pos, ray_to, physicsClientId=self.client)
            hit_id, hit_fraction = ray_test[0][0], ray_test[0][2]
            
            if hit_id == self.target_id:
                radar_dists.append(1.0)
            elif hit_id in self.all_enemy_ids:
                radar_dists.append(-hit_fraction)
            else:
                radar_dists.append(hit_fraction)
        return radar_dists

    def get_camera_obs(self):
        if self.drone_id is None: return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.float32)
        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        tilt_rad = np.radians(self.camera_tilt_deg)
        forward = rot_matrix.dot([1.0, 0.0, 0.0])
        downward = rot_matrix.dot([0.0, 0.0, -1.0])
        cam_target = np.array(pos) + (forward * np.cos(tilt_rad) + downward * np.sin(tilt_rad))
        cam_up = rot_matrix.dot([0.0, 0.0, 1.0])

        view_matrix = p.computeViewMatrix(cameraEyePosition=list(pos), cameraTargetPosition=cam_target.tolist(), cameraUpVector=cam_up.tolist(), physicsClientId=self.client)
        proj_matrix = p.computeProjectionMatrixFOV(fov=self.camera_fov, aspect=float(self.camera_width)/self.camera_height, nearVal=self.camera_near, farVal=self.camera_far, physicsClientId=self.client)
        images = p.getCameraImage(self.camera_width, self.camera_height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
        return (np.reshape(images[2], (self.camera_height, self.camera_width, 4))[:, :, :3].astype(np.float32) / 255.0)

    def step(self, action=None):
        if self.drone_crashed:
            p.stepSimulation(physicsClientId=self.client)
            return True

        speed = 6.0
        vel = np.array([action[0]*speed, action[1]*speed, action[2]*speed]) if action is not None else np.array([0.0, 0.0, 0.0])

        if self.level >= 3:
            self._update_wind()
            vel += self.wind_direction * self.wind_strength * 0.12
            self._update_enemy_behavior()
            self._update_ground_radars()

        p.resetBaseVelocity(self.drone_id, linearVelocity=vel.tolist(), physicsClientId=self.client)
        self.update_radar()
        self._check_collisions()
        p.stepSimulation(physicsClientId=self.client)
        return self.drone_crashed

    def close(self):
        p.disconnect(self.client)