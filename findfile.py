import tkFileDialog
import tkMessageBox
import os
import last_directory

def default_selection_fun(x):
    return x.startswith('20') or x.startswith('21') or x.startswith('punkt')

def sorting_fun_by_os_stat(x,y):
    def filetime(filename):
        # I think it will always be st_mtime, but dont know this. todo look it up
        s = os.stat(filename)
        t = min([s.st_atime, s.st_mtime, s.st_ctime])
        return t
    tx = filetime(x)
    ty = filetime(y)
    if tx == ty:
        return 0
    else:
        return -1 if tx<ty else 1


def default_sorting_fun(x,y):
    #sort by name!
    # sorting_fun_by_os_stat is very slow because os.stat is slow.
    # always now saving the files with a name beginning with the date and time
    if x==y:
        return 0
    return -1 if x<y else 1

def path_slash(name):
    return name.replace('\\','/')

def my_path_join(path,name):
    return path_slash(os.path.join(path,name))

def repath(filename_with_folder):
    folder, name = os.path.split(filename_with_folder)
    return my_path_join(folder,name)

class File_list(object):
    def __init__(self, filetypes, save_last_filename,selection_fun=None,
                 sorting_fun = None):
        self.filetypes = filetypes
        self.file_selected = None
        self.opened_nr = 0
        fun = selection_fun
        self.selection_fun = default_selection_fun if fun is None else fun
        fun = sorting_fun
        self.sorting_fun = default_sorting_fun if fun is None else fun 
        self.lastdir = last_directory.remember(save_last_filename)
        self.current = None
        self.file_selected = None
        self.nfiles = 0
        self.index = 0
        #thefiles = os.listdir('d:/temp/ncycle_gent_2014_/results')
        #thefiles = [os.path.join('d:/temp/ncycle_gent_2014_/results', x)
        #            for x in thefiles if x.startswith('201')];
        #self.file_selected = thefiles[10]
    def findfile(self, file = None):
        if file is None:
            file = tkFileDialog.askopenfilename(initialdir=self.lastdir.get())
        file = repath(file)
        # filetypes = [('dat','.dat')])
        self.file_selected = file;
        self.current = file;
        self.update()
        self.lastdir.set(os.path.split(file)[0])
        return file
    def update(self):
        katalog, name = os.path.split(self.file_selected)
        all_files = os.listdir(katalog)
        #ext = os.path.splitext(name)[1]
        all_files = [x for x in all_files if self.selection_fun(x)]
        self.all_files = [my_path_join(katalog,x) for x in all_files]
        self.all_files.sort(cmp=self.sorting_fun)
        self.index = self.all_files.index(path_slash(self.current))
        self.nfiles = len(self.all_files)
    def next(self):
        if self.index>=len(self.all_files)-1:
            self.update()
        if self.index<len(self.all_files)-1:
            self.index += 1
            self.current = self.all_files[self.index]
        return self.current
    def previous(self):
        if self.index>0:
            self.index -= 1
            self.current = self.all_files[self.index]
        return self.current
    def first(self):
        self.index = 0
        self.current = self.all_files[0]
        return self.current
    def last(self):
        self.index = len(self.all_files)-1
        self.current = self.all_files[self.index]
        return self.current
