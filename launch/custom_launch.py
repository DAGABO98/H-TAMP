import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the path to the Navigation2 bringup launch file
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_launch_file = os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
    
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={
            # Pass in any required launch arguments for Navigation2 here.
            'use_sim_time': 'false',
            # You can add additional parameters as needed.
        }.items()
    )

    # Optionally, include your custom node as an ExecuteProcess or another IncludeLaunchDescription.
    custom_node = ExecuteProcess(
        cmd=['ros2', 'run', 'my_nav2_pkg', 'custom_node'],
        output='screen'
    )

    return LaunchDescription([
        nav2_launch,
        custom_node
    ])

