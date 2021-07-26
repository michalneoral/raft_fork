from setuptools import setup, find_packages

print(find_packages())

setup(
    name='RAFT',
    version='0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'pypng',
        'h5py',
        'jupyter',
        'notebook',
        'opencv-contrib-python>=4.0.0',
        'opencv-python>=4.0.0',
        'Pillow',
        'scipy',
        'tqdm',
        'scikit-learn'
        ]#,

)
