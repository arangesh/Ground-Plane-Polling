import setuptools

setuptools.setup(
    name='keras-retinanet-3D',
    description='Keras implementation of RetinaNet 3D object detector.',
    author='Akshay Rangesh',
    author_email='arangesh@ucsd.edu',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet', 'six', 'scipy'],
    entry_points = {
        'console_scripts': [
            'retinanet-3D-train=keras_retinanet_3D.bin.train:main',
            'retinanet-3D-debug=keras_retinanet_3D.bin.debug:main',
            'retinanet-3D-convert-model=keras_retinanet_3D.bin.convert_model:main',
            'retinanet-3D-run-network=keras_retinanet_3D.bin.run_network:main',
        ],
    }
)
