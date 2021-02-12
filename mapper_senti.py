#!/usr/bin/env python

import sys

for line in sys.stdin:
     keyval=line.strip().split("\t")
     key,val=(keyval[0],keyval[0])
     if key!="\N" and val!="\N":
        sys.stdout.write('%s\t%s\n' % (key, val))
