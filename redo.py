import subprocess
import sys
import os
path=sys.argv[1]
if not os.path.isdir("summary"):
    subprocess.call(['mkdir','summary'])
subprocess.call(['python','convnet.py',path])
