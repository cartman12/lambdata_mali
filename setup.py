""" 
Naive Majority Classifier - Works on categorical features only

"""

from distutils.core import setup
setup(
  name = 'lambdata_mali_classifier',         
  packages = ['lambdata_mali_classifier'],   # Chose the same as "name"
  version = '0.3',      # 0.1 is the first version
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository  
  description = 'This library is a ML module that predicts using a naive majority Decision Tree',
  author = 'Mohamad Ali Nasser',                   
  author_email = 'mhd.ali.nasser@gmail.com',      # E-Mail
  url = 'https://github.com/cartman12/lambdata_mali_classifier',   
  download_url = 'https://github.com/cartman12/lambdata_mali_classifier/archive/v0.3.tar.gz',   
  keywords = ['ML', 'Decision Tree', 'Majority Classifier'],   # Keywords that define your package best
  install_requires=[ 'pandas' ,        
          'sklearn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3.7',
  ],
)