import os
import sys
import subprocess
import re
import threading
import pprint


def putfirst(item, lst):
    if item in lst:
        lst.insert(0, lst.pop(lst.index(item)))
    return lst


def my_prints(x, maxn=5, levels=4, truncate=100, indent=0, do_ind=1):
    def rest(maxn):
        return(maxn[:1] if len(maxn) == 1 else maxn[1:])
    if isinstance(maxn, int):
        maxn = [maxn]
    s = ''

    def prt(x, indent, end='\n'):
        if not isinstance(x, str):
            x = repr(x)
        if len(x) > truncate:
            x = x[:truncate] + '...'
        return ' ' * indent + x + end
    if isinstance(x, (int, float)):
        s += prt(x, indent * do_ind)
    elif isinstance(x, (list, tuple)):
        if all([isinstance(y, (int, float)) for y in x[:maxn[0]]]):
            if len(x) > maxn[0]:
                if isinstance(x, list):
                    s += prt(x[:maxn[0]] + ['...'], indent)
                else:
                    s += prt(x[:maxn[0]] + ('...',), indent)
            else:
                s += prt(x, indent)
        else:
            s += prt('(' if isinstance(x, tuple) else '[', indent * do_ind, '')
            for i in range(min(len(x), maxn[0])):
                s += my_prints(x[i], rest(maxn), levels,
                               indent + levels, i > 0)
            if len(x) > maxn[0]:
                s += prt('...', indent)
            s = s[:-1]
            s += prt(')' if isinstance(x, tuple) else ']', 0)
    elif isinstance(x, dict):
        s += '\n' + prt('{', indent, end='')
        for i, (key, value) in enumerate(x.items()):
            s += prt(repr(key) + ':', indent + 1 if i else 0, end='')
            s += my_prints(value, rest(maxn), levels, indent + levels)
        s = s.rstrip('\n')
        s += '}\n'
    else:  # isinstance(x, str):
        s += prt(x, indent * do_ind)
    return s


def my_print(x, maxn=5, levels=2, truncate=100):
    print(my_prints(x, maxn, levels, truncate))
    sys.stderr.flush()
    sys.stdout.flush()


def myhook(value):
    import __main__
    if value is not None:
        __main__.__builtins__._ = value
        print('ok')
        pprint.pprint(value)


class Myprint:
    maxn = 5
    levels = 4
    truncate = 100


def myhook2(value):
    import __main__
    if value is not None:
        __main__.__builtins__._ = value
        my_print(value,
                 maxn=Myprint.maxn,
                 levels=Myprint.levels,
                 truncate=Myprint.truncate)


def myhook3(value):
    import __main__
    if value is not None:
        __main__.__builtins__._ = value
        s = str(value)
        if len(s) > 10000:
            s = s[:9990] + ' (...) ' + s[-5:]
        print(s)


if repr(sys.displayhook).startswith('<built-in'):
    orig_displayhook = sys.displayhook


def myprint_on(use_pprint=False, maxn=None, levels=None, truncate=None):
    """
if use_pprint is True, use Myprint to set maxn, levels and truncate
if maxn, levels and/or truncate is given, this sets 
Myprint.maxn, Myprint.levels and/or Myprint.truncate """
    if use_pprint >= 0:
        setattr(sys, 'displayhook', myhook)
    elif use_pprint == -1:
        setattr(sys, 'displayhook', myhook3)
    else:
        setattr(sys, 'displayhook', myhook2)


def myprint_off():
    setattr(sys, 'displayhook', orig_displayhook)

# myprint_on(-1)

#__main__.__builtins__.pprint_on = lambda: setattr(sys, 'displayhook', myhook)
#__main__.__builtins__.pprint_off = lambda: setattr(sys, 'displayhook', orig_displayhook)


def make_fieldname(s):
    if isinstance(s, str):
        if s[0] in '0123456789':
            s = '_' + s
        s = re.sub("[\[\]\(\)-+,.:' \t]", '_', s)
    elif isinstance(s, int):
        s = make_fieldname(repr(s))
    return s


class Empty_class():
    pass


def dict2inst(x):  # just to be able to use tab key on the repl to show dicts..
    if isinstance(x, dict):
        a = Empty_class()
        for key, v in x.items():
            setattr(a, make_fieldname(key), dict2inst(v))
    elif isinstance(x, (list, tuple)):
        a = [dict2inst(xi) for xi in x]
    else:
        a = x
    if isinstance(x, tuple):
        a = tuple(a)
    return a


def catch_dos_command_output(command, filename='_slettmeg.txt'):
    if isinstance(command, str):
        command = command.split()
    subprocess.call(command + ['>', '_slettmeg.txt'], shell=True)
    with open('_slettmeg.txt') as f:
        return f.read()


def get_com_ports(filter=None):
    q = catch_dos_command_output('wmic path Win32_SerialPort')
    a = str(q, "utf_16le")[1:].split('\r\n')
    if filter is None:
        return a
    else:
        return [x for x in a if x.find(filter) >= 0]


def find_com_nr(strings):
    if not strings:
        return None
    i = strings[0].find('(COM')
    i2 = strings[0].find(')', i)
    return str(strings[0][i + 1:i2])  # str converts from unicode


def find_arduino():
    return find_com_nr(get_com_ports('Arduino Uno'))


def find_relay_card():
    return find_com_nr(get_com_ports('K8090'))


def trun(fun, *args, **kwargs):
    daemon = kwargs.pop('_daemon') if '_daemon' in kwargs else False
    tr = threading.Thread(target=fun, args=args, kwargs=kwargs)
    if daemon:
        tr.daemon = True
    tr.start()
    return tr


def colwrite(sv, n, number=False):
    if number:
        sv = [repr(i) + ' ' + str(x) for i, x in enumerate(sv)]
    rows = []
    nrows = len(sv) / n
    if len(sv) > n * nrows:
        nrows += 1
    for i in range(nrows):
        rows.append(sv[i::nrows])
    maxn = [0] * n
    for r in rows:
        for i, e in enumerate(r):
            if len(str(e)) > maxn[i]:
                maxn[i] = len(str(e))
    for r in rows:
        s = ''
        for n, e in zip(maxn, r):
            s += str(e).ljust(n + 3)
        print(s)


def named_tuplify(dct, name='NT', recursive=1000):
    from collections import namedtuple
    if (not isinstance(dct, dict)) or not recursive:
        return dct
    keys = list(dct.keys())
    A = namedtuple(name, keys)
    return(A(*[named_tuplify(dct[key], recursive=recursive - 1) for key in keys]))


def raise_error_if_not_exist(dir_or_filename):
    
    if not os.path.exists(dir_or_filename):
        raise Exception('%s does not exist' % dir_or_filename)
  

def ensure_absolute_path(name, newbase='.', maybe_new=True):
    """If name is not an absolute path, return [newbase]/name. newbase may
    be absolute or relative to current path.  Raises error if
    resulting path does not exist, except for, if maybe_new is True,
    the last part of the path/filename
    newbase may be a list or tuple which will be joined by os.path.join.
    """
    if isinstance(newbase, (list, tuple)):
        newbase = os.path.join(*newbase)
    if not os.path.isabs(newbase):
        newbase = os.path.join(os.getcwd(),
                               newbase if newbase != '.' else '')
    if not os.path.isabs(name):
        name = os.path.join(newbase, name)
    if maybe_new:
        parent = os.path.split(name)[0]
        raise_error_if_not_exist(parent)
    else:
        raise_error_if_not_exist(name)
    return os.path.normpath(name)
        
