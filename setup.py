from setuptools import setup, find_packages

setup(
    name='UTIS-HeliostatBeamCharacterization',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.2',                
        'Pillow>=10.4.0',                
        'pytorch_lightning>=2.1.4',    
        'torch>=2.0.0',                
        'torchvision>=0.15.0',         
    ],
    entry_points={
        'console_scripts': [
            'train-model=utis.train:main',      # Exposes the training script
            'run-inference=tests.predict:main',  # Exposes the inference script
        ],
    },
    author='Mathias Kuhl',
    author_email='mathias.kuhl@dlr.de',
    description='UNet-based Target Image Segmentation for Solar Tower Calibration',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DLR-SF/UTIS-HeliostatBeamCharacterization',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
