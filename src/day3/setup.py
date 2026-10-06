from setuptools import find_packages, setup

package_name = 'day3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hv-06',
    maintainer_email='alekdi8gm30@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "nav_to_pose = day3.3_1_a_nav_to_pose:main",
            "nav_through_poses = day3.3_1_b_nav_through_poses:main",
            "follow_waypoints = day3.3_1_c_follow_waypoints:main",
            "create_path = day3.3_1_d_create_path:main",
            "mail_delivery = day3.3_1_e_mail_delivery:main",
            "patrol_loop = day3.3_1_f_patrol_loop:main",
            "sim_depth_checker = day3.3_2_c_sim_depth_checker:main",
            "sim_depth_to_3d = day3.3_2_b_sim_depth_to_3d:main",
            "sim_depth_to_nav_goal = day3.3_2_c_sim_depth_to_nav_goal:main",
            "depth_checker = day3.3_3_a_depth_checker:main",
            "depth_to_3d = day3.3_3_b_depth_to_3d:main",
            "depth_to_3d_ts = day3.3_3_b_depth_to_3d_ts:main",
            "depth_to_nav_goal = day3.3_3_c_depth_to_nav_goal:main",
            "depth_to_nav_goal_ts = day3.3_3_d_depth_to_nav_goal_ts:main",
        ],
    },
)
