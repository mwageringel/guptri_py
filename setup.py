from numpy.distutils.core import Extension
import numpy.distutils.command.build_ext
import distutils
from distutils.command import clean
from distutils.core import Command
from distutils import log
from distutils.dir_util import mkpath, remove_tree
import os
import sys

# original Guptri website went offline
# UPSTREAM_TAR_URL = ("https://www8.cs.umu.se/research/nla/singular_pairs/guptri/archive/guptri.tar.gz",
#                     "https://www8.cs.umu.se/research/nla/singular_pairs/guptri/archive/fguptri.tar.gz")
UPSTREAM_TAR_URL = ("https://web.archive.org/web/20210310205556/https://www8.cs.umu.se/research/nla/singular_pairs/guptri/archive/guptri.tar.gz",
                    "https://web.archive.org/web/20210310204056/https://www8.cs.umu.se/research/nla/singular_pairs/guptri/archive/fguptri.tar.gz")
CHECKSUMS = ("42ba92fb3b58334d99b43e0e4338caeeba91d8d8", "3416bed23a0bf4175bec3e1579821c217c29a7bb")
UPSTREAM_TAR_NAMES = ("guptri.tar.gz", "fguptri.tar.gz")

cwd = os.path.abspath(os.getcwd())
UPSTREAM_DIR = os.path.join(cwd, "upstream")
SRC = "guptri_py"
VERSION = open("VERSION").read().strip()
TMPDIR = "local/var/tmp"

def sha1sum(filename):
    import hashlib
    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    with open(filename, 'rb') as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BLOCKSIZE)
    return hasher.hexdigest()

class DownloadCommand(Command):
    description = "download, extract and patch guptri files"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        log.info("Checking if upstream tar files exist.")
        if not os.path.exists(UPSTREAM_DIR):
            mkpath(UPSTREAM_DIR)
        for tarname, checksum, url in zip(UPSTREAM_TAR_NAMES, CHECKSUMS, UPSTREAM_TAR_URL):
            t_file = os.path.join(UPSTREAM_DIR, tarname)
            if not os.path.exists(t_file):
                log.warn("File %s does not exist. Attempting to download from %s." % (t_file, url))
                os.system("wget -O %s %s" % (t_file, url))
                if not os.path.exists(t_file):
                    log.error("""Download failed. You may wish to download the file "%s" manually from "%s" and place it in the "upstream/" directory.""" %
                            (tarname, url))
                    sys.exit(1)
            if sha1sum(t_file) != checksum:
                log.error("Checksum for file %s is different." % t_file)
                sys.exit(1)

        log.info("Creating directories.")
        if os.path.exists("local"):
            remove_tree("local")
        mkpath(TMPDIR)

        log.info("Extracting tar files.")
        for tarname in UPSTREAM_TAR_NAMES:
            t_file = os.path.join(UPSTREAM_DIR, tarname)
            import tarfile
            tar = tarfile.open(t_file)
            tar.extractall(TMPDIR)
            tar.close()

        log.info("Applying patches.")
        if os.system("patch -d %s < patches/0001-fix-compile-errors-fguptri.patch" % TMPDIR):
            log.error("Failed to apply patches.")
            sys.exit(1)

class BuildExtCommand(numpy.distutils.command.build_ext.build_ext):
    def run(self):
        self.run_command('download')
        super().run()

class CleanCommand(clean.clean):
    def run(self):
        if os.path.exists("local"):
            remove_tree("local")
        super().run()

class TestCommand(Command):
    description = "run Sage tests (after installation)"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        errno = os.system("sage -t --force-lib %s" % SRC)
        if errno != 0:
            sys.exit(1)

ext1 = Extension(name='guptri_py._fguptri_py',
                 # the following silences type mismatch errors in gfortran 10
                 # (-fallow-argument-mismatch is not recognized by older fortran versions)
                 extra_f77_compile_args=['-std=legacy'],
                 sources=['guptri_py/_fguptri_py.pyf'] +
                         [os.path.join(TMPDIR, s) for s in ('fguptri.f', 'guptribase.f', 'zguptri.f')])

from numpy.distutils.core import setup
setup(
    cmdclass={
        'download': DownloadCommand,
        'build_ext': BuildExtCommand,
        'test': TestCommand,
        'clean': CleanCommand,
        },
    name='guptri_py',
    version=VERSION,
    packages=["guptri_py"],
    ext_modules=[ext1])
