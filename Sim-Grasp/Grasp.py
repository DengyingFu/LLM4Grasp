
import genesis as gs
import torch
import random
import time
import cv2
import math
import numpy as np
import open3d as o3d
import requests
import base64
from matplotlib import pyplot as plt
from mpmath import euler
from open3d.examples.geometry.triangle_mesh_transformation import scale
from openai import OpenAI
from graspnet.graspnet import GraspBaseline
import copy
import re
from scipy.spatial.transform import Rotation as R
import open3d as o3d

'''
    该版本改变了技能API的命名。
    使用时需要修改DINO 还是YOLO，传入的数据不同格式
'''
class EmboidedGrasp:
    def __init__(self):
        self.graspNet = GraspBaseline()  # 加载抓取检测模型
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 3, 0.8),
                camera_lookat=(0.0, 0.8, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, -10.0),
                substeps=4,  # 增加子步骤提高抓取稳定性
            ),
            show_viewer=True,
            show_FPS=False
        )
        self.scene.add_entity(gs.morphs.Plane())  # 添加地面
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(
                file='xml/franka_emika_panda/panda.xml',
                pos = (0, 0, 0),
                euler = (0, 0, 90),  # scipy外旋x-y-z,单位度
                # quat = (1.0, 0.0, 0.0, 0.0), # w-x-y-z四元数
                scale = 1.0,
        ),
        )
        self.load_objects()

        camera = self.setup_capture_camera() #添加相机
        self.scene.build()
        self.videocam.start_recording()
        camera.set_pose(self.transform)
        self.extrinsics = camera.extrinsics
        print(self.transform)
        print('外参矩阵')
        print(camera.extrinsics)
        print('外参矩阵的逆')
        print(np.linalg.inv(self.extrinsics))

        # 定义关节索引
        self.motors_dof = np.arange(7)  # 机械臂关节
        self.fingers_dof = np.arange(7, 9)  # 夹爪关节

        self.set_Control_parameter()
        self.move_ee([0., 0.1, 0.8] + [np.pi, 0, 0], 'end')  # 机械臂移动到初始位置 是按xyz顺序转动的
        # self.move_ee([-0.3, 0.6, 0.1] + [np.pi, 0, 0], 'end')
        torch.set_printoptions(precision=6, sci_mode=False)
        degree = np.degrees([np.pi, 0, 0])  # 弧度rpy转角度xyz

        self.move_gripper('O')
        for i in range(200):
            self.scene.step()
        color, depth, segmentation, normal = self.cam.render(depth=False, segmentation=False, normal=False)
        color, depth, segmentation, normal = self.cam.render(depth=False, segmentation=False, normal=False)
        self.save_jpg(color, "1")
        self.initial_state = self.scene.get_state()

        #测试=====
        # grasp_poses = self.get_grasp()
        # grasp = self.filter_grasp_by_text('apple', grasp_poses)  # 过滤
        # self.execute_grasp(grasp[0])
        # self.ungrasp()
        # self.execute_grasp(grasp)
        # xyzl = self.get_put('purple cup')
        # self.execute_putit(xyzl)
        # self.videocam.stop_recording(save_to_filename='test.mp4', fps=60)
        #=======
        # while 1:
        #     self.scene.step()

    def load_objects(self):
        # self.screws = self.scene.add_entity(gs.morphs.Mesh(file="/home/dyfu/DataSet/Traceparts/1/screw_fybox-nkj2-3.stl",
        #                                                   pos=(0, 0.65, 0.2),
        #                                                   euler=(0, 90, 0),
        #                                                   scale=0.01
        #                                                   ))
        # self.screws2 = self.scene.add_entity(gs.morphs.Mesh(file="/home/dyfu/DataSet/Traceparts/2/screw_fybcbl4-10.stl",
        #                                                    pos=(0.3, 0.65, 0.2),
        #                                                    euler=(0, 0, 0),
        #                                                    scale=0.01
        #                                                    ))
        # self.screws3 = self.scene.add_entity(gs.morphs.Mesh(file="/home/dyfu/DataSet/Traceparts/3/nuts_fysntrc4.stl",
        #                                                    pos=(-0.3, 0.65, 0.2),
        #                                                    euler=(90, 0, 0),
        #                                                    scale=0.01
        #                                                    ))
        """Part-Net可用：3558红瓶子，3616黄瓶子，103723刀"""
        self.table = self.scene.add_entity(gs.morphs.URDF(file="./ycb_objects/table/table.urdf",
                                                          pos=(0, 0.65, 0.05),
                                                          euler=(0, 0, 0),
                                                          scale=1.1,
                                                          collision=True
                                                          ))
        self.knife = self.scene.add_entity(gs.morphs.URDF(file="/home/dyfu/DataSet/PartNet-mobility/103723/mobility.urdf",
                                                          pos=(-0.0, 0.7, 0.2),
                                                          euler=(0, 0, 0),
                                                          scale=0.1))
        self.bowl = self.scene.add_entity(gs.morphs.Mesh(file="./ycb_objects/green_bowl/textured.obj",
                                                          # pos=(-0.45, 0.55, 0.2),
                                                         pos=(-0.2, 0.6, 0.2),
                                                          euler=(0, 0, 0),
                                                          scale=1.0,
                                                          convexify=False,
                                                         decompose_nonconvex=True
                                                          ))

        self.apple = self.scene.add_entity(
            gs.morphs.URDF(file='/home/dyfu/DataSet/GraspNet-1B/urdfs/012.urdf',
                           pos=(0.1, 0.6, 0.2),
                           euler=(0, 0, 0),
                           scale=0.8
                           )
        )
 
        self.j_cups = self.scene.add_entity(
            gs.morphs.Mesh(file='/home/dyfu/DataSet/ycb_urdfs/ycb_assets/065-f_cups/google_16k/textured.obj',
                           pos=(-0.3, 0.4, 0.2),
                           euler=(0, 0, 0),
                           scale=1.2,
                           convexify=False,
                           )
        )

        self.yellowbox = self.scene.add_entity(gs.morphs.URDF(file='/home/dyfu/DataSet//GraspNet-1B/urdfs/001.urdf',
                                                        # pos=(-0.3, 0.6, 0.2),
                                                              pos=(0.35, 0.6, 0.2),
                                                        euler=(0,90,0),
                                                        scale=0.6,
                                                            )
                                             )
        # self.redcan = self.scene.add_entity(
        #     gs.morphs.URDF(file='/home/dyfu/DataSet/GraspNet-1B/urdfs/002.urdf',
        #                    pos=(0.3, 0.6, 0.25),
        #                    euler=(90, 0, 0),
        #                    scale=0.65
        #                    )
        # )


        # self.banana = self.scene.add_entity(
        #     gs.morphs.Mesh(file='/home/dyfu/DataSet/GraspNet-1B/005/textured.obj',
        #                    pos=(-0.1, 0.55, 0.2),
        #                    euler=(0, 0, 0),
        #                    scale=1.2,
        #                    convexify=False,
        #                    )
        # )
        # self.banana = self.scene.add_entity(
        #     gs.morphs.URDF(file='/home/dyfu/DataSet/GraspNet-1B/urdfs/005.urdf',
        #                    pos=(-0.1, 0.55, 0.2),
        #                    euler=(0, 0, 0),
        #                    scale=1.1,
        #                    )
        # )


    def set_Control_parameter(self):
        # 设置控制器参数
        # 注意：以下值是为实现Franka最佳行为而调整的。
        # 有时高质量的URDF或XML文件也会提供这些参数，并会被解析。
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        end_effector = self.franka.get_link('hand')  # 获取末端执行器
        # IK控制（逆运动学）
        if control_method == 'end':
            print(action)
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            degree = np.degrees([roll, pitch, yaw])  # 弧度rpy转角度xyz
            ori = gs.xyz_to_quat(torch.tensor(degree))  # 欧拉角(度数) 转为 四元数
            # 计算需要的关节角度
            self.joint_poses = self.franka.inverse_kinematics(
                link=end_effector,
                pos=np.array(pos),
                quat=np.array(ori),
            )

            # 规划运动路径
            path = self.franka.plan_path(
                qpos_goal=self.joint_poses,
                num_waypoints=400,  # 2秒时长
                ignore_joint_limit=True,
                ignore_collision=True
            )
            for waypoint in path:
                self.franka.control_dofs_position(waypoint[:-2], self.motors_dof) #不控制最后两个关节，最后两个是夹爪
                self.scene.step()
                self.videocam.render()

    def move_gripper(self, statu):
        if statu=='C':
            # self.franka.control_dofs_force(np.array([-10, -10]), self.fingers_dof)  # 控制夹爪关节
            self.franka.control_dofs_position(np.array([0,0]), self.fingers_dof)
        elif statu=='O':
            print('松开')
            self.franka.control_dofs_position(np.array([0.12,0.12]), self.fingers_dof) #控制夹爪关节
            # self.franka.control_dofs_force(np.array([0, 0]), self.fingers_dof)  # 控制夹爪关节
        pass


    def setup_capture_camera(self):
        # 设置采集摄像头参数
        self.width = 640
        self.height = 480

        self.fov = 100
        self.aspect = self.width / self.height
        self.near = 0.02
        self.far = 1

        self.camera_pos = [0., 0.65, 0.4]
        self.target_Pos = [0., 0.65, 0.0]
        self.up_vector = [0, 1, 0]

        self.transform = gs.pos_lookat_up_to_T(self.camera_pos, self.target_Pos, self.up_vector)  # 相机在世界坐标系的位置
        self.cam = self.scene.add_camera(
            res=(self.width, self.height),
            pos=self.camera_pos,
            lookat=self.target_Pos,
            up=self.up_vector,
            fov=self.fov,
            GUI=True,
            # transform=self.transform,
        )
        self.videocam = self.scene.add_camera(
            res=(960, 640),
            pos=[0,3,1.5],
            lookat=(0.0, 0.8, 0.5),
            up=[0,-1,0],
            fov=30,
            GUI=False,
            # transform=self.transform,
        )
        self.intrinsics = self.cam.intrinsics
        print('内参矩阵')
        print(self.intrinsics)
        return self.cam

    def rgbd2points(self):
        color, depth, segmentation, normal = self.cam.render(depth=True, segmentation=False, normal=False)
        color, depth, segmentation, normal = self.cam.render(depth=True, segmentation=False, normal=False)

        color_r = color[..., 0].reshape(-1)
        color_g = color[..., 1].reshape(-1)
        color_b = color[..., 2].reshape(-1)
        # print(color.shape)
        # print(depth.shape)
        # print(color)
        # print(depth)
        # 生成实际像素坐标网格
        u = np.arange(self.width)
        v = np.arange(self.height)
        u_grid, v_grid = np.meshgrid(u, v)
        u_flat = u_grid.reshape(-1)
        v_flat = v_grid.reshape(-1)
        z = depth.reshape(-1)

        # 过滤无效深度
        mask = z < 0.99
        u_valid = u_flat[mask]
        v_valid = v_flat[mask]
        z_valid = z[mask]

        # 构造齐次像素坐标 [3, N]
        pixel_coords = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)
        pixel_coords = pixel_coords * z_valid  #这里乘了下面就不用乘了
        # 转换到相机坐标系
        points_camera_homogeneous = np.linalg.inv(self.intrinsics) @ pixel_coords  #3*N，其中每一列是相机坐标系中的方向向量
        # points_camera = points_camera_homogeneous * z_valid  # 乘以深度
        # points_camera_homogeneous[:,:1] *= -1

        rgbs = np.stack([color_r, color_g, color_b], axis=1).astype('float32') / 255.0
        rgbs = rgbs[z < 0.99]
        # 转置为[N, 3]
        self.points_camera = points_camera_homogeneous.T
        self.rgbs = rgbs
        self.depth = depth
        self.color = color

    def grasp2pixel(self, grasp_poses): #将world坐标系下的3维点云投影到像素坐标系
        grasp_points = []
        for grasp in grasp_poses['grasp_world']:
            grasp_point = grasp[:, -1].reshape((-1, 4)) #1*4
            grasp_points.append(grasp_point)
        #得到变换矩阵的最后一列
        grasp_points = np.concatenate(grasp_points, axis=0).T # 4*N ,N 个 grasp
        # 这里乘内外参矩阵直接转换到了实际像素坐标系
        grasp_camera = self.extrinsics @ grasp_points #world 2 camera
        grasp_screen = self.intrinsics @ grasp_camera[:3, :] #camera 2 pixel
        grasp_screen = grasp_screen / grasp_camera[2, :] #归一化到pixel实际像素坐标系

        return grasp_screen[:2, :].T
    #============fdy============

    def pixel2grasp(self, pixel_coords, depth_img):
        """
        将像素坐标转换为世界坐标系下的坐标。
        步骤：
            先通过4x4的投影矩阵算出相机内参矩阵参数, 然后得到深度, 最后通过相似三角形的关系得到相机坐标系的xyz。
            然后通过视角矩阵计算出外参矩阵，转换到世界坐标系下
        参数:
            pixel_coords (list or np.ndarray): 像素坐标列表，形状为 (N, 2)，每行为 (x, y)。
            depth_img (np.ndarray): 深度图，形状为 (H, W)。

        返回:
            np.ndarray: 世界坐标点，形状为 (N, 4)。最后一列是其次项1
        """
        # 1. 获取必要的矩阵
        # projection_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        # view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")

        # 2. 从投影矩阵提取相机内参参数
        f_x = self.intrinsics[0][0]
        f_y = self.intrinsics[1][1]
        c_x = self.intrinsics[0][2]
        c_y = self.intrinsics[1][2]

        # 3. 将像素坐标转换到相机坐标系
        camera_coords = []
        for x, y in pixel_coords:
            # 从深度图获取深度值
            # Z_c = self.far * self.near / (self.far - (self.far - self.near) * depth_img[int(y), int(x)])  # Bullet中相机坐标下的深度值
            Z_c = depth_img[int(y), int(x)]# 相机坐标下的深度值
            # 由相似三角形反投影到相机坐标系
            X_c = (x - c_x) * Z_c / f_x
            Y_c = (y - c_y) * Z_c / f_y

            camera_coords.append([X_c, Y_c, Z_c])
        camera_coords = np.asarray(camera_coords)

        # 4. 将相机坐标转换为齐次坐标
        camera_coords_homogeneous = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])  # (N, 4)

        # 5. 应用视角矩阵逆矩阵计算camera2world的外参矩阵(绕x轴旋转180度) 渲染引擎的坐标系约定不同
        # Tc = np.array([[1,  0,  0,  0],
        #            [0,  -1,  0,  0],
        #            [0,  0,  -1,  0],
        #            [0,  0,  0,  1]]).reshape(4,4)
        # Vt = np.linalg.inv(self.extrinsics) @ Tc
        Vt = np.linalg.inv(self.extrinsics)
        # 6. 将相机坐标转换到世界坐标系
        world_coords_homogeneous = Vt @ camera_coords_homogeneous.T
        world_coords_homogeneous = world_coords_homogeneous.T
        world_coords_homogeneous[0][2] = self.camera_pos[2]-Z_c+0.3
        return world_coords_homogeneous

    def save_jpg(self, color, name):
        tmp_image_path = name + '.jpg'
        cv2.imwrite(tmp_image_path, color[..., :3][..., ::-1])

    def detect_by_text_img(self, color, class_text):
        tmp_image_path = 'tmp.jpg'
        cv2.imwrite(tmp_image_path, color[..., :3][..., ::-1]) # 保存RGB图像，以供后续调用yolo-world进行开放词汇检测
        # 读取图片并转换为 Base64 编码
        with open(tmp_image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 准备请求数据
        #Yolo
        # data = {
        #     'image': image_base64,
        #     'classes': [class_text]
        # }
        #DINO
        data = {
            'image': image_base64,
            'classes': class_text
        }
        # 发送 POST 请求
        response = requests.post('http://127.0.0.1:5000/detect', json=data)
        # response = requests.post('http://10.104.2.60:5000/detect', json=data)  #DINO 部署在服务器
        # 处理响应
        if response.status_code == 200:
            detections = response.json().get('Detections', [])
            print('Detections:', detections)
            return detections
        else:

            print('Error:', response.text)
            return []

    def detect_by_text(self, class_text):
        virtual_color = self.color
        tmp_image_path = 'tmp.jpg'
        cv2.imwrite(tmp_image_path, virtual_color[..., :3][..., ::-1]) # 保存RGB图像，以供后续调用yolo-world进行开放词汇检测
        # 读取图片并转换为 Base64 编码
        with open(tmp_image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 准备请求数据
        #YOLO
        # data = {
        #     'image': image_base64,
        #     'classes': [class_text]
        # }
        #DINO
        data = {
            'image': image_base64,
            'classes': class_text
        }
        # 发送 POST 请求
        response = requests.post('http://127.0.0.1:5000/detect', json=data)
        # response = requests.post('http://10.104.2.60:5000/detect', json=data)  #DINO 部署在服务器
        # 处理响应
        if response.status_code == 200:
            detections = response.json().get('Detections', [])
            print('Detections:', detections)
            return detections
        else:

            print('Error:', response.text)
            return []

    def filter_grasp_by_text(self, text, grasp_poses, vis=True): # 根据2d检测bbox，对grasp_poses进行filter
        detections = self.detect_by_text(text)
        grasp_screen = self.grasp2pixel(grasp_poses)
        if len(detections):
            det = detections[0]
            x1, y1, x2, y2, score, class_name = det

            grasp_candidate = []
            for i, grasp_p in enumerate(grasp_screen):
                # 根据grasp_points是否在bbox进行filter
                if grasp_p[0]>x1 and grasp_p[0]<x2 and grasp_p[1]>y1 and grasp_p[1]<y2:
                    grasp_candidate.append(i)

            if len(grasp_candidate):

                scores = [grasp_poses['score'][i] for i in grasp_candidate]
                grasp_points = [grasp_screen[i] for i in grasp_candidate]
                grasps = [grasp_poses['grasp_world'][i] for i in grasp_candidate]

                # 简单规则，更偏向垂直方向的grasp
                grasps_rule_score = np.array([abs(grasp_poses['grasp_world'][i][2, 0]) for i in grasp_candidate])
                max_id = np.argmax(np.array(scores) + grasps_rule_score)

                # max_id = np.argmax(np.array(scores))
                if vis:
                    plt.imshow(self.color)

                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red'))
                    plt.gca().text(x1, y1 - 5, f'{score:.2f}', color='red', fontsize=10, backgroundcolor='none')

                    plt.plot(grasp_points[max_id][0], grasp_points[max_id][1], 'go')
                    # plt.title(f'{class_text} Grasp Score: {scores[max_id]:.2f}')
                    plt.gca().set_axis_off()  # 关闭整个坐标轴（包括背景）
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除四周留白
                    plt.savefig("2.jpg", bbox_inches='tight', pad_inches=0)
                    plt.show()


                    # 创建可视化窗口
                    viewer = o3d.visualization.Visualizer()
                    viewer.create_window()

                    # 添加原始点云（转换到世界坐标系）
                    pcd_world = o3d.geometry.PointCloud()
                    pcd_world.points = o3d.utility.Vector3dVector(self.points_camera)
                    pcd_world.colors = o3d.utility.Vector3dVector(self.rgbs)

                    pcd_world.transform(np.linalg.inv(self.extrinsics))  # 转换到世界坐标系
                    viewer.add_geometry(pcd_world)

                    # 添加所有候选抓取姿态（蓝色）
                    for i, grasp_idx in enumerate(grasp_candidate):
                        grasp_mesh = self.gg[grasp_idx].to_open3d_geometry()
                        grasp_mesh.paint_uniform_color([0, 0.5, 1])  # 蓝色
                        grasp_mesh.transform(np.linalg.inv(self.extrinsics))  # 转换到世界坐标系
                        viewer.add_geometry(grasp_mesh)

                    # 高亮最佳抓取姿态（绿色）
                    best_grasp_mesh = self.gg[grasp_candidate[max_id]].to_open3d_geometry()
                    best_grasp_mesh.paint_uniform_color([0, 1, 0])  # 绿色
                    best_grasp_mesh.transform(np.linalg.inv(self.extrinsics))  # 转换到世界坐标系
                    viewer.add_geometry(best_grasp_mesh)


                    # 设置视角参数
                    ctr = viewer.get_view_control()
                    ctr.set_front([0, -1, 0.5])  # 设置相机朝向
                    ctr.set_up([0, 0, 1])  # 设置垂直方向
                    ctr.set_zoom(0.8)  # 缩放级别

                    viewer.run()
                    viewer.destroy_window()
                return [grasps[max_id]] #返回存有grasp矩阵的列表

        return []

    def get_grasp(self, samples=1000, vis=True):
        self.rgbd2points()  # 根据rgb和的depth得到点云图像
        pcd = o3d.geometry.PointCloud()  #摄像头坐标系下的点云坐标
        # 将NumPy数组设置为点云的点
        pcd.points = o3d.utility.Vector3dVector(self.points_camera)
        pcd.colors = o3d.utility.Vector3dVector(self.rgbs)

        # pybullet camera to world
        trans_camera_world = self.extrinsics #view_matrix是world to camera,取逆后是camera to world

        def get_grasp_world(grasp):
            translation = grasp.translation
            rot = grasp.rotation_matrix
            grasp_trans = np.eye(4)  #grasp矩阵
            grasp_trans[:3, :3] = rot
            grasp_trans[:3, -1] = translation
            # grasp_trans_world = np.linalg.inv(trans_camera_world) @ axis_correction @ grasp_trans
            # 将grasp转换到世界坐标系
            grasp_trans_world = np.linalg.inv(trans_camera_world).dot(grasp_trans) #grasp->camera->world
            return grasp_trans_world

        gg = self.graspNet.run(copy.deepcopy(pcd), vis=False) #进行检测

        gg.nms()
        gg.sort_by_score()

        grasp_poses = {'score':[], 'grasp_world':[]}

        for grasp in gg[:samples]: #按score取前samples个预测抓取结果，遍历
            # 获取grasp在world坐标下的变换
            grasp_world = get_grasp_world(grasp) #抓取姿态从graspnet坐标转回到world坐标，graspnet坐标和相机坐标系一样
            grasp_poses['score'].append(grasp.score)
            grasp_poses['grasp_world'].append(grasp_world)

        if vis:
            viewer = o3d.visualization.Visualizer()
            viewer.create_window()
            viewer.add_geometry(pcd.transform(np.linalg.inv(trans_camera_world)))# 点云乘外参矩阵的逆，转回到世界坐标

            for grasp in gg[:samples]:
                mesh_grasp = grasp.to_open3d_geometry().transform(np.linalg.inv(trans_camera_world))# grasp矩阵乘外参矩阵的逆，转回到世界坐标
                viewer.add_geometry(mesh_grasp)

            viewer.run()
            viewer.destroy_window()
        self.gg = gg
        return grasp_poses  #返回得分前samples个抓取姿态矩阵，字典


    def get_put(self, class_text,vis=True):
        '''
        输入：
            得到需要放置到的位置，返回 加入了虚拟视角
        返回：
            None
        '''
        # color, depth, segmentation, normal = self.cam.render(depth=True, segmentation=False, normal=False)
        # color, depth, segmentation, normal = self.cam.render(depth=True, segmentation=False, normal=False)
        # self.color = color
        detections = self.detect_by_text_img(self.color, class_text)
        if len(detections):
            det = detections[0]
            x1, y1, x2, y2, score, class_name = det
            #得到检测框中心坐标
            x = x1 + (x2 - x1)/2.0
            y = y1 + (y2 - y1)/2.0
            #转换到世界坐标
            xyz1 = self.pixel2grasp([[x,y]], self.depth)  #得到变换矩阵的最后一列，二维数组
            # xyz1[0][2] += 0.05  #抬高一点避免触碰到底
            res = xyz1[0].tolist()
            if vis:
                plt.imshow(self.color)

                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red'))
                # plt.gca().text(x1, y1 - 5, f'{score:.2f}', color='red', fontsize=10, backgroundcolor='none')
                plt.plot(x1+(x2-x1)/2, y1+(y2-y1)/2, 'gx')
                # plt.title(f'{class_text} ')
                plt.gca().set_axis_off()  # 关闭整个坐标轴（包括背景）
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除四周留白
                plt.savefig("3.jpg", bbox_inches='tight', pad_inches=0)
                plt.show()
            # 渲染虚拟视角图像，重新检测目标
            translation = np.array([-res[0], res[1], 0.5])  # X 方向平移
            T_new = np.eye(4)
            T_new[:3,:3] = self.extrinsics[:3,:3]
            T_new[:3,3] = translation
            new_img = self.project_to_new_view(T_new, self.intrinsics, self.width, self.height)
            if vis:
                plt.imshow(new_img)
                plt.show()
            detections = self.detect_by_text_img(new_img, class_text)  # yolo检测目标位置（结果是像素坐标）
            if len(detections):
                det = detections[0]
                x1, y1, x2, y2, score, class_name = det
                # 得到检测框中心坐标
                x = x1 + (x2 - x1) / 2.0
                y = y1 + (y2 - y1) / 2.0
                # 转换到世界坐标
                xyz1 = self.pixel2grasp([[x, y]], self.depth)  # 得到变换矩阵的最后一列，二维数组
                # xyz1[0][2] += 0.1  # 抬高一点避免触碰到底
                res2 = xyz1[0].tolist()
                T_new = np.eye(4)
                T_new[:3, 3] = np.array([res[0], res[1]-0.65, 0.])
                res2 = T_new @ res2
                if vis:
                    plt.imshow(new_img)
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red'))
                    # plt.gca().text(x1, y1 - 5, f'{score:.2f}', color='red', fontsize=10, backgroundcolor='none')
                    plt.plot(x1+(x2-x1)/2, y1+(y2-y1)/2, 'gx')
                    # plt.title(f'{class_text} ')
                    # plt.savefig("4.jpg")
                    plt.gca().set_axis_off()  # 关闭整个坐标轴（包括背景）
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除四周留白
                    plt.savefig("4.jpg", bbox_inches='tight', pad_inches=0)
                    plt.show()
                return res2
            return res
        else:  #没有检测到目标位置的话，随便搞一个位置
            return [0.5,0.5,0.8,1]

    def execute_putit(self, xyz1):
        """
            放置到指定位置
        """
        self.move_xyz(xyz1[:3])
        print('松开：')
        self.ungrasp()
        xyz1[2] += 0.05
        self.move_xyz(xyz1[:3])

    def execute_grasp(self, grasp_world):
        # pre grasp offset，以供后续夹爪进行抓取
        trans_x_neg = np.eye(4)
        trans_x_neg[0, -1] = -0.2 #离物体有点距离
        pre_grasp_world1 = grasp_world.dot(trans_x_neg)

        def get_xyzrpy(grasp): #夹爪坐标系和graspnet中的夹爪坐标系有区别 https://github.com/graspnet/graspnet-baseline/issues/65
            # print('===============旋转前后======================')
            # print(R.from_matrix(grasp[:3, :3]).as_euler("yzx"))
            # 作y轴旋转
            R2 = R.from_euler("zyx", [0, np.pi / 2, 0], degrees=False)
            T2 = np.eye(4, 4)
            T2[:3, :3] = R2.as_matrix()
            grasp = np.dot(grasp, T2)
            # print(R.from_matrix(grasp[:3, :3]).as_euler("yzx"))

            rot = R.from_matrix(grasp[:3, :3])
            euler = rot.as_euler('xyz')
            xyzrpy = np.array(grasp[:3, -1].tolist()+euler.tolist())
            return xyzrpy

        grasp_location1 = get_xyzrpy(pre_grasp_world1)
        xyz = grasp_location1[:3].tolist()
        # self.move_xyz(xyz)

        self.move_xyzrpy(grasp_location1)

        trans_x_neg[0, -1] = -0.08  #再接近物体一点
        pre_grasp_world2 = grasp_world.dot(trans_x_neg)
        grasp_location2 = get_xyzrpy(pre_grasp_world2)
        self.move_xyzrpy(grasp_location2)
        # self.move_xyz(grasp_location[:3].tolist())

        self.grasp() #闭合夹爪 抓取

        self.move('up',0.2)
        # self.move_gripper_down()
        # self.move_xyzrpy(grasp_location1)
        # self.move_xyz(xyz)
        pass

    def move(self, direction, step=0.1):
        """
        控制机械臂末端执行器沿指定方向移动指定步长。

        :param direction: 移动方向，可选值："up", "down", "front", "back", "left", "right"
        :param step: 移动步长，正值表示向指定方向移动，负值表示反向移动。单位是厘米cm
        """
        # 获取末端执行器的当前位姿
        pos = self.franka.get_links_pos().cpu().numpy()[8]
        quat = self.franka.get_links_quat().cpu().numpy()[8]
        eluer = np.deg2rad(gs.quat_to_xyz(quat))

        # 方向映射
        if direction == "up":
            pos[2] += step
        elif direction == "down":
            pos[2] -= step
        elif direction == "front":
            pos[1] += step
        elif direction == "back":
            pos[1] -= step
        elif direction == "left":
            pos[0] -= step
        elif direction == "right":
            pos[0] += step
        else:
            raise ValueError("Invalid direction. Choose from: 'up', 'down', 'front', 'back', 'left', 'right'")

        self.move_ee(list(pos) + list(eluer), 'end')

        for i in range(200):
            self.scene.step()
            self.videocam.render()

    def move_to(self,xyz1):
        self.move_xyz(xyz1[:3])

    def move_reset(self):
        self.move_ee([0., 0.1, 0.8] + [np.pi, 0, 0], 'end')
        for i in range(200):
            self.scene.step()
            self.videocam.render()

    def move_gripper_down(self):
        # 获取末端执行器的当前位姿
        pos = self.franka.get_links_pos().cpu().numpy()[8]
        self.move_ee(list(pos)+[np.pi,0,0],'end')

    def move_gripper_up(self):
        # 获取末端执行器的当前位姿
        pos = self.franka.get_links_pos().cpu().numpy()[8]
        self.move_ee(list(pos)+[0,0,0],'end')

    def move_xyz(self, xyz):
        self.move_ee(list(xyz)+[np.pi, 0, 0], 'end')
        for i in range(500):
            self.scene.step()
            self.videocam.render()
            # time.sleep(1/240)


    def move_xyzrpy(self, xyzrpy):
        self.move_ee(xyzrpy, 'end') #再转动夹爪
        for i in range(300):
            self.scene.step()
            self.videocam.render()
            # time.sleep(1/240)

    def ungrasp(self):
        self.move_gripper('O')
        for i in range(300):
            self.scene.step()
            self.videocam.render()

    def grasp(self):
        self.move_gripper('C')
        for i in range(300):
            self.scene.step()
            self.videocam.render()

    def capture_from_new_view(self, camera_pos, target_pos, up_vector):
        """设置相机新视角并捕获RGB图像"""
        new_transform = gs.pos_lookat_up_to_T(camera_pos, target_pos, up_vector)
        self.cam.set_pose(new_transform)
        self.extrinsics = self.cam.extrinsics
        for _ in range(10):
            self.scene.step()
            self.videocam.render()
        color, _, _, _ = self.cam.render(depth=False, segmentation=False, normal=False)
        color, _, _, _ = self.cam.render(depth=False, segmentation=False, normal=False)
        return color

    def pointcloud_to_rgb_image(self):
        # 获取相机内参
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        # 点云和颜色数据
        points = self.points_camera  # 形状为 (N, 3)
        colors = self.rgbs  # 形状为 (N, 3)，范围在0-1之间

        # 分解坐标
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        # 计算投影后的像素坐标（浮点数）
        u = (fx * X / Z) + cx
        v = (fy * Y / Z) + cy

        # 转换为整数像素坐标
        u_ints = np.round(u).astype(int)
        v_ints = np.round(v).astype(int)

        # 过滤超出图像边界的坐标
        valid_mask = (u_ints >= 0) & (u_ints < self.width) & (v_ints >= 0) & (v_ints < self.height)
        u_valid = u_ints[valid_mask]
        v_valid = v_ints[valid_mask]
        Z_valid = Z[valid_mask]
        colors_valid = colors[valid_mask]

        # 初始化深度缓冲区和RGB图像
        depth_buffer = np.full((self.height, self.width), np.inf)
        rgb_image = np.zeros((self.height, self.width, 3))

        # 处理每个有效点，更新最近点的颜色
        for i in range(len(u_valid)):
            u_pix = u_valid[i]
            v_pix = v_valid[i]
            z = Z_valid[i]
            color = colors_valid[i]

            if z < depth_buffer[v_pix, u_pix]:
                depth_buffer[v_pix, u_pix] = z
                rgb_image[v_pix, u_pix] = color

        # 转换颜色范围到0-255并转为uint8类型
        rgb_image = (rgb_image * 255).astype(np.uint8)
        cv2.imwrite("output_image.jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        return rgb_image


    def project_to_new_view(self, T_new, K_new, new_width, new_height):
        """得到虚拟视角图像"""
        # 转换点云到新相机坐标系
        points_original = self.points_camera  # [N, 3]
        N = points_original.shape[0]
        # 转换为齐次坐标并应用变换矩阵
        homogeneous_points = np.hstack((points_original, np.ones((N, 1))))
        # print(homogeneous_points.shape)
        homogeneous_points = (np.linalg.inv(self.extrinsics) @ homogeneous_points.T).T
        points_new_homo = (T_new @ homogeneous_points.T).T
        points_new = points_new_homo[:, :3]

        # 新相机内参分解
        fx, fy = K_new[0, 0], K_new[1, 1]
        cx, cy = K_new[0, 2], K_new[1, 2]

        # 提取坐标和深度，过滤无效点
        X, Y, Z = points_new[:, 0], points_new[:, 1], points_new[:, 2]
        valid = Z > 0
        X, Y, Z = X[valid], Y[valid], Z[valid]
        colors = self.rgbs[valid]

        # 计算投影后的像素坐标
        u = (fx * X / Z) + cx
        v = (fy * Y / Z) + cy
        u_idx = np.round(u).astype(int)
        v_idx = np.round(v).astype(int)

        # 过滤越界坐标
        in_bounds = (u_idx >= 0) & (u_idx < new_width) & (v_idx >= 0) & (v_idx < new_height)
        u_idx, v_idx = u_idx[in_bounds], v_idx[in_bounds]
        Z, colors = Z[in_bounds], colors[in_bounds]

        # 深度缓冲初始化
        depth_buffer = np.full((new_height, new_width), np.inf)
        color_buffer = np.zeros((new_height, new_width, 3))
        # color_buffer = np.ones((new_height, new_width, 3))

        # 处理每个有效点，更新颜色和深度
        for i in range(len(u_idx)):
            ui, vi, zi = u_idx[i], v_idx[i], Z[i]
            if zi < depth_buffer[vi, ui]:
                depth_buffer[vi, ui] = zi
                color_buffer[vi, ui] = colors[i]

        # 转换为uint8图像
        new_image = (color_buffer * 255).astype(np.uint8)
       
        return new_image


if __name__=="__main__":
    grasp_system = EmboidedGrasp()
    while True:
        grasp_system.scene.step()


