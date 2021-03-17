from distutils.core import setup
setup(
  name = 'RcTorch',         # How you named your package folder (MyLib)
  packages = ['RcTorch'],   # Chose the same as "name"
  version = '0.62',      # Start with a small number and increase it with every change you make
  license='Harvard',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = '',   # Give a short description about your library
  author = 'Hayden Joy',                   # Type in your name
  author_email = 'hnjoy@mac.com',      # Type in your E-Mail
  url = 'https://github.com/blindedjoy/RcTorch',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/blindedjoy/RcTorch/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Echo State Network', 'ESN', 'Reservoir Computing', 'Echo State Networks', 'Optimization', 'BoTorch', 'PyTorch', 'Bayesian'],   # Keywords that define your package best
  description='A Python 3 toolset for creating and optimizing Echo State Networks. This library is an extension and expansion of the previous library written by Reinier Maat: https://github.com/1Reinier/Reservoir',
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
          'paramz',
          'datetime',
          'pandas',
          'seaborn',
          'pathlib',
          'matplotlib',
          'torch',#==1.7.1',
          'torchvision',#==0.8.2',
          'torchaudio', ##==0.7.2',
          'gpytorch',
          'botorch'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.6',
  ],

)