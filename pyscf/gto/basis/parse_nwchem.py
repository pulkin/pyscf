#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# parse NWChem format
#

MAXL = 8
SPDF = ('S', 'P', 'D', 'F', 'G', 'H', 'I', 'J')
MAPSPDF = {'S': 0,
           'P': 1,
           'D': 2,
           'F': 3,
           'G': 4,
           'H': 5,
           'I': 6,
           'J': 7}

def parse(string):
    '''Parse the basis text which is in NWChem format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`
    Lines started with # are ignored.
    '''
    bastxt = []
    for dat in string.splitlines():
        x = dat.split('#')[0].strip().upper()  # Use # to start comments
        if (x and not x.startswith('END') and not x.startswith('BASIS')):
            bastxt.append(x)
    return _parse(bastxt)

def load(basisfile, symb):
    return _parse(search_seg(basisfile, symb))

def parse_ecp(string):
    ecptxt = []
    for dat in string.splitlines():
        x = dat.split('#')[0].strip().upper()
        if (x and not x.startswith('END') and not x.startswith('ECP')):
            ecptxt.append(x)
    return _parse_ecp(ecptxt)

def load_ecp(basisfile, symb):
    return _parse_ecp(search_ecp(basisfile, symb))

def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:
        # ignore head
        dat = fin.readline().upper()
        while dat and not dat.strip()[1:].lstrip().startswith('BASIS'):
            dat = fin.readline().upper()

        # searching
        dat = fin.readline().upper()
        while dat and not dat.lstrip().startswith('END'):
            dat = dat.strip()
            if dat.split(' ', 1)[0] == symb.upper():
                seg = []
                while dat:
                    dat = dat.strip()
                    if (dat[1:].lstrip().startswith('BASIS') or
                        dat.startswith('END')):
                        break
                    seg.append(dat)
                    dat = fin.readline().upper()
                return seg
            else:
                while dat and not dat.strip()[1:].lstrip().startswith('BASIS'):
                    dat = fin.readline().upper()
            dat = fin.readline().upper()
    return []

def search_ecp(basisfile, symb):
    with open(basisfile, 'r') as fin:
        # ignore head
        dat = fin.readline().upper()
        while dat and not dat.lstrip().startswith('ECP'):
            dat = fin.readline().upper()

        dat = fin.readline().upper()
        # searching
        while dat:
            dat = dat.strip()
            if (dat.split(' ', 1)[0] == symb.upper() or
                dat.startswith('END')):
                break
            dat = fin.readline().upper()

        seg = []
        while dat:
            dat = dat.strip()
            if ((dat[0].isalpha() and dat.split(' ', 1)[0] != symb.upper()) or
                dat.startswith('END')):
                return seg
            elif dat: # remove blank lines
                seg.append(dat)
            dat = fin.readline().upper()
    return []

def convert_basis_to_nwchem(symb, basis):
    '''Convert the internal basis format to NWChem format string'''
    from pyscf.gto.mole import _std_symbol
    res = []
    symb = _std_symbol(symb)

    # pass 1: comment line
    ls = [b[0] for b in basis]
    nprims = [len(b[1:]) for b in basis]
    nctrs = [len(b[1])-1 for b in basis]
    prim_to_ctr = {}
    for i, l in enumerate(ls):
        if l in prim_to_ctr:
            prim_to_ctr[l][0] += nprims[i]
            prim_to_ctr[l][1] += nctrs[i]
        else:
            prim_to_ctr[l] = [nprims[i], nctrs[i]]
    nprims = []
    nctrs = []
    for l in set(ls):
        nprims.append(str(prim_to_ctr[l][0])+SPDF[l].lower())
        nctrs.append(str(prim_to_ctr[l][1])+SPDF[l].lower())
    res.append('#BASIS SET: (%s) -> [%s]' % (','.join(nprims), ','.join(nctrs)))

    # pass 2: basis data
    for bas in basis:
        res.append('%-2s    %s' % (symb, SPDF[bas[0]]))
        for dat in bas[1:]:
            res.append(' '.join('%15.9f'%x for x in dat))
    return '\n'.join(res)

def convert_ecp_to_nwchem(symb, ecp):
    '''Convert the internal ecp format to NWChem format string'''
    from pyscf.gto.mole import _std_symbol
    symb = _std_symbol(symb)
    res = ['%-2s nelec %d' % (symb, ecp[0])]

    for ecp_block in ecp[1]:
        l = ecp_block[0]
        if l == -1:
            res.append('%-2s ul' % symb)
        else:
            res.append('%-2s %s' % (symb, SPDF[l].lower()))
        for r_order, dat in enumerate(ecp_block[1]):
            for e,c in dat:
                res.append('%d    %15.9f  %15.9f' % (r_order, e, c))
    return '\n'.join(res)

def _parse(raw_basis):
    basis_add = []
    for line in raw_basis:
        dat = line.strip()
        if dat.startswith('#'):
            continue
        elif dat[0].isalpha():
            key = dat.split()[1]
            if key == 'SP':
                basis_add.append([0])
                basis_add.append([1])
            else:
                basis_add.append([MAPSPDF[key]])
        else:
            line = [float(x) for x in dat.replace('D','e').split()]
            if key == 'SP':
                basis_add[-2].append([line[0], line[1]])
                basis_add[-1].append([line[0], line[2]])
            else:
                basis_add[-1].append(line)
    bsort = []
    for l in range(MAXL):
        bsort.extend([b for b in basis_add if b[0] == l])
    return bsort

def _parse_ecp(raw_ecp):
    ecp_add = []
    nelec = None
    for line in raw_ecp:
        dat = line.strip()
        if dat.startswith('#'): # comment line
            continue
        elif dat[0].isalpha():
            key = dat.split()[1]
            if key == 'NELEC':
                nelec = int(dat.split()[2])
                continue
            elif key == 'UL':
                ecp_add.append([-1])
            else:
                ecp_add.append([MAPSPDF[key]])
            by_ang = [[], [], [], []]
            ecp_add[-1].append(by_ang)
        else:
            line = dat.replace('D','e').split()
            l = int(line[0])
            by_ang[l].append([float(x) for x in line[1:]])
    if nelec is not None:
        bsort = []
        for l in range(-1, MAXL):
            bsort.extend([b for b in ecp_add if b[0] == l])
        return [nelec, bsort]

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O', basis='6-31g')
    print(load_ecp('lanl2dz.dat', 'Na'))
    b = load('ano.dat', 'Na')
    print(convert_basis_to_nwchem('Na', b))
    b = load_ecp('lanl2dz.dat', 'Na')
    print(convert_ecp_to_nwchem('Na', b))
