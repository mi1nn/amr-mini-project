from setuptools import find_packages, setup

package_name = 'rokey_pjt'

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
            'depth_checker = rokey_pjt.depth_checker:main',
            'depth_checker_mouse = rokey_pjt.depth_checker_mouse:main',
            'depth_check_claude = rokey_pjt.depth_check_claude:main',
            'yolo_detector = rokey_pjt.yolo_detector:main',
            'calibrate_webcam = rokey_pjt.calibrate_webcam:main',
            'rc_car_follower = rokey_pjt.rc_car_follower:main',
            'webcam_publisher = rokey_pjt.webcam_publisher:main',
        ],
    },
)
