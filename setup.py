"""
The onion package contains the OpenPathSampling python library
to do transition interface sampling.

"""
from setuptools import setup
import os
import subprocess

##########################
VERSION = "0.1.0"
ISRELEASED = False
__version__ = VERSION
##########################


################################################################################
# Writing version control information to the module
################################################################################

def git_version():
    # Return the git revision as a string
    # copied from numpy setup.py
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


def write_version_py(filename='onion/version.py'):
    cnt = """
# This file is automatically generated by setup.py
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()

################################################################################
# Installation
################################################################################

write_version_py()


def build_keyword_dictionary():
    setup_keywords = {}
    setup_keywords["name"]              = "onion"
    setup_keywords["version"]           = "0.0.1"
    setup_keywords["author"]            = "Jan-Hendrik Prinz, Josh Fass and John D. Chodera"
    setup_keywords["author_email"]      = "jan.prinz@choderalab.org, josh.fass@choderalab.org, choderaj@mskcc.org"
    setup_keywords["license"]           = "LGPL 2.1 or later"
    setup_keywords["url"]               = "http://github.com/choderalab/onion"
    setup_keywords["download_url"]      = "http://github.com/choderalab/onion"
    setup_keywords["packages"]          = ['onion'
                                          ]
    setup_keywords["package_dir"]       = {
        'onion': 'onion'
    }
    setup_keywords["data_files"]        = []
    setup_keywords["ext_modules"]       = []
    setup_keywords["platforms"]         = ["Linux", "Mac OS X", "Windows"]
    setup_keywords["description"]       = "Tools for automatic TIS Interface discovery."
    setup_keywords["requires"]          = ["numpy", "nose"]
    setup_keywords["long_description"]  = """
    Onion (http://github.com/choderalab/onion) is a python library to automatically determine transition interfaces from MSMs.
    """
    outputString = ""
    firstTab     = 40
    secondTab    = 60
    for key in sorted( setup_keywords.iterkeys() ):
         value         = setup_keywords[key]
         outputString += key.rjust(firstTab) + str( value ).rjust(secondTab) + "\n"
    
    print("%s" % outputString)

    return setup_keywords
    

def main():
    setup_keywords = build_keyword_dictionary()
    setup(**setup_keywords)

if __name__ == '__main__':
    main()




