from distutils.core import setup
setup(
  name = 'GPadversarial',
  packages = ['GPadversarial'], # this must be the same as the name above
  version = '1.01',
  description = 'A python module for generating adversarial samples for GP classifiers',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/GPadversarial.git',
  download_url = 'https://github.com/lionfish0/GPadversarial/archive/1.01.tar.gz',
  keywords = ['adversarial examples','gaussian processes'],
  classifiers = [],
  install_requires=['GPy','numpy','matplotlib'],
)
