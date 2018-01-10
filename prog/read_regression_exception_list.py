import xlrd


def parse_xls_file(filename):
    book = xlrd.open_workbook(filename)
    rows = list(book.sheet_by_index(0).get_rows())
    start = None
    for i, r in enumerate(rows):
        if r[0].value == 'START':
            start = i+1
    if start is None:
        raise Exception("""I can't find the start of the list. There should be a cell in the
        first column containting the word START. In the next row there
        shold be the headings: Use, Filename, interval etc. Consult a
        valid exception xlsx file""")
    rows = rows[start:]
    headings = [x.value for x in rows[0][2:]]
    ret = {}
    for i, r in enumerate(rows[1:]):
        if not r[0].value:
            continue
        filename = r[1].value
        options = {}
        for j, elt in enumerate(r[2:]):
            if elt.value != '':
                # we need to take a value like 10 and a heading like
                # 'left:CO2:start' and turn it into
                # {'left': {'CO2': {'start' : 10}}}
                p = options
                keys = headings[j].split(':')
                for k in keys[:-1]:
                    if k not in p:
                        p[k] = {}
                    p = p[k]
                p[keys[-1]] = elt.value
        ret[filename] = options
    return ret

