import os, sys

class remember(object):
    """ usage:
    import last_directory
    lastdir = last_directory.remember(file='test.txt',folder = None)
    print lastdir.get()
    lastdir.set('c:/temp')
    print lastdir.get()"""
    def __init__(self,filename, folder = None):
        if folder is None:
            folder = os.path.realpath(os.path.dirname(sys.argv[0]))
        last_directory_file = os.path.join(folder, filename)
        if not os.path.exists(last_directory_file):
            if not os.path.isdir(os.path.split(last_directory_file)[0]):
                raise RuntimeError(last_directory_file + ' Directory does not exist.')
            else:
                fid = open(last_directory_file,'w');
                fid.close()#lager fila hvis den ikke finnes
        self.last_directory_file = last_directory_file
    def get(self):
        try:
            fid=open(self.last_directory_file,'r')
            katalog=fid.readline().strip()
            if not os.path.exists(katalog):
                katalog=''#'c:/'
                fid.close()
        except Exception:
            katalog = ''
        return katalog
    def set(self,directory):
            fid=open(self.last_directory_file,'w')
            fid.write(directory)
            fid.close()
