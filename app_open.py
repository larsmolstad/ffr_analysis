def open(filepath):
    import subprocess, os, sys
    if sys.platform.startswith('darwin'):
       subprocess.call(('open', filepath))
    elif os.name == 'nt':
    	 os.startfile(filepath)
    elif os.name == 'posix':
    	 subprocess.call(('xdg-open', filepath))
