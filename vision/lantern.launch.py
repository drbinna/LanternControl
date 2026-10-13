"""Lantern vision stack: D455 infrared stereo -> Isaac VSLAM -> PX4 bridge."""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

HERE = os.path.dirname(os.path.realpath(__file__))


def generate_launch_description():
    camera = Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        namespace='camera', name='camera', output='screen',
        parameters=[{
            'enable_infra1': True,
            'enable_infra2': True,
            'enable_color': False,
            'enable_depth': False,
            'depth_module.emitter_enabled': 0,      # IR dots confuse VSLAM
            'depth_module.infra_profile': '640x360x90',
            'enable_gyro': False,                   # D455 IMU not detected
            'enable_accel': False,
        }])

    vslam = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,
            'enable_imu_fusion': False,
            'image_jitter_threshold_ms': 22.0,
            'base_frame': 'camera_link',
            'camera_optical_frames': ['camera_infra1_optical_frame',
                                      'camera_infra2_optical_frame'],
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', '/camera/camera/infra1/camera_info'),
            ('visual_slam/image_1', '/camera/camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', '/camera/camera/infra2/camera_info'),
        ])

    container = ComposableNodeContainer(
        name='visual_slam_launch_container', namespace='',
        package='rclcpp_components', executable='component_container_mt',
        composable_node_descriptions=[vslam], output='screen')

    # Bridge + watchdog; restarts itself if it ever stops.
    bridge = ExecuteProcess(
        cmd=['python3', os.path.join(HERE, 'vslam_to_px4.py')],
        output='screen', respawn=True, respawn_delay=1.0)

    return LaunchDescription([camera, container, bridge])
