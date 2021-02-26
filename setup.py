from setuptools import setup, find_packages

print(find_packages())

setup(
    name='raft', 
    version='0.1',
    description='',
    packages=find_packages(),
    # dependency_links=[
    #     'https://download.pytorch.org/whl/torch_stable.html',
    #     ],
    install_requires=[
        # 'torch==1.5.0+cu101',
        # 'torchvision==0.6.0+cu101',
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
    # entry_points={
    #     'console_scripts': [
    #         '',
    #     ]}
)
