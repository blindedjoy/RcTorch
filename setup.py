from distutils.core import setup
#from setuptools import setup, Extension

#read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

#this_file = Path(__file__).parent
#print(this_directory)
#long_description = this_file#.read_text()



setup(
  name = 'rctorch',         # How you named your package folder (MyLib)
  packages = ['rctorch'],   # Chose the same as "name"
  version = '0.91',      # Start with a small number and increase it with every change you make
  license='Harvard',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description='A Python 3 toolset for creating and optimizing Echo State Networks. This library is an extension and expansion of the previous library written by Reinier Maat: https://github.com/1Reinier/Reservoir',# Give a short description about your library
  author = 'Hayden Joy',                   # Type in your name
  author_email = 'hnjoy@mac.com',      # Type in your E-Mail
  url = 'https://github.com/blindedjoy/RcTorch',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/blindedjoy/RcTorch/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Echo State Network', 'ESN', 'Reservoir Computing', 'Echo State Networks', 'Optimization', 'BoTorch', 'PyTorch', 'Bayesian'],   # Keywords that define your package best
  requirements = [ "botorch==0.5.1",
				   "GPy==1.10.0",
				   "GPyOpt==1.2.6",
				   "gpytorch==1.5.1",
				   "matplotlib==3.4.2",
				   "numpy==1.22.3",
				   "pandas==1.3.4",
				   "paramz==0.9.5",
				   "ray==1.13.0",
				   "scikit_learn==1.0.2",
				   "scipy==1.7.1",
				   "seaborn==0.11.2",
				   "setuptools==58.4.0",
				   "torch==1.10.0",
				   "ipython==7.26.0",
					 ],
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
          'paramz',
          #'datetime',
          'pandas',
          'seaborn',
          'pathlib',
          'matplotlib',
          #'llvmlite',
          'torch',
          'torchvision',
          'torchaudio',
          'gpytorch',
          'botorch',
          #'ray' #[default]
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
 #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.9',
  ],
  long_description=long_description,
  long_description_content_type='text/markdown'

)

