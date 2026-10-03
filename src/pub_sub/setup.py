from setuptools import find_packages, setup

package_name = 'pub_sub'

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
    description='Basic ROS 2 string and image publisher/subscriber examples',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_publisher = pub_sub.data_publisher:main',
            'data_subscriber = pub_sub.data_subscriber:main',
            'image_publisher = pub_sub.image_publisher:main',
            'image_subscriber = pub_sub.image_subscriber:main',
        ],
    },
)
