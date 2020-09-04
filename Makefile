# This Makefile is for convenience as a reminder and shortcut for the most used commands

# Package folder
PACKAGE=guptri_py

# change to your sage command if needed
SAGE=sage
# SAGE=python

PIP=$(SAGE) -pip

all: install test

install:
	$(PIP) install --upgrade --no-index -v .

install-user:
	$(PIP) install --upgrade --no-index -v --user .

uninstall:
	$(PIP) uninstall $(PACKAGE)

develop:
	$(PIP) install --upgrade -e .

test:
	$(SAGE) setup.py test

coverage:
	$(SAGE) -coverage $(PACKAGE)/*

doc:
	cd docs && $(SAGE) -sh -c "make html"

# doc-pdf:
# 	cd docs && $(SAGE) -sh -c "make latexpdf"

clean: doc-clean
	rm -rf local/

doc-clean:
	cd docs && rm -rf _build/

pyf:
	#sage -python -m numpy.f2py fguptri.f -m guptri_py/_fguptri_py -h guptri_py/_fguptri_py.pyf
	$(SAGE) -python -m numpy.f2py -c $(PACKAGE)/_fguptri_py.pyf local/var/tmp/fguptri.f local/var/tmp/guptribase.f local/var/tmp/zguptri.f

.PHONY: all install install-user uninstall develop test coverage clean doc-clean doc doc-pdf pyf
