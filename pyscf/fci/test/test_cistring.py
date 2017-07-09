#!/usr/bin/env python

import unittest
import numpy
from pyscf.fci import cistring


class KnowValues(unittest.TestCase):
    def test_strings4orblist(self):
        ref = ['0b1010', '0b100010', '0b101000', '0b10000010', '0b10001000',
               '0b10100000']
        for i, x in enumerate(cistring.gen_strings4orblist([1,3,5,7], 2)):
            self.assertEqual(bin(x), ref[i])
        ref = ['0b11', '0b101', '0b110', '0b1001', '0b1010', '0b1100',
               '0b10001', '0b10010', '0b10100', '0b11000']
        for i, x in enumerate(cistring.gen_strings4orblist(range(5), 2)):
            self.assertEqual(bin(x), ref[i])

    def test_linkstr_index(self):
        idx1 = cistring.gen_linkstr_index_o0(range(4), 2)
        idx2 = cistring.gen_linkstr_index(range(4), 2)
        idx23 = numpy.array([[0, 0, 3, 1],
                             [3, 3, 3, 1],
                             [1, 0, 4, 1],
                             [2, 0, 5, 1],
                             [1, 3, 0, 1],
                             [2, 3, 1, 1],])
        self.assertTrue(numpy.all(idx1[:,:,2:] == idx2[:,:,2:]))
        self.assertTrue(numpy.all(idx23 == idx2[3]))

        idx1 = cistring.gen_linkstr_index(range(7), 3)
        idx2 = cistring.reform_linkstr_index(idx1)
        idx3 = cistring.gen_linkstr_index_trilidx(range(7), 3)
        idx3[:,:,1] = 0
        self.assertTrue(numpy.all(idx2 == idx3))

    def test_addr2str(self):
        self.assertEqual(bin(cistring.addr2str(6, 3, 7)), '0b11001')
        self.assertEqual(bin(cistring.addr2str(6, 3, 8)), '0b11010')
        self.assertEqual(bin(cistring.addr2str(7, 4, 9)), '0b110011')

    def test_str2addr(self):
        self.assertEqual(cistring.str2addr(6, 3, int('0b11001' ,2)), 7)
        self.assertEqual(cistring.str2addr(6, 3, int('0b11010' ,2)), 8)
        self.assertEqual(cistring.str2addr(7, 4, int('0b110011',2)), 9)

    def test_gen_cre_str_index(self):
        idx = cistring.gen_cre_str_index(range(4), 2)
        idx0 = [[[ 2, 0, 0, 1], [ 3, 0, 1, 1]],
                [[ 1, 0, 0,-1], [ 3, 0, 2, 1]],
                [[ 0, 0, 0, 1], [ 3, 0, 3, 1]],
                [[ 1, 0, 1,-1], [ 2, 0, 2,-1]],
                [[ 0, 0, 1, 1], [ 2, 0, 3,-1]],
                [[ 0, 0, 2, 1], [ 1, 0, 3, 1]]]
        self.assertTrue(numpy.allclose(idx, idx0))

    def test_gen_des_str_index(self):
        idx = cistring.gen_des_str_index(range(4), 2)
        idx0 = [[[ 0, 0, 1,-1], [ 0, 1, 0, 1]],
                [[ 0, 0, 2,-1], [ 0, 2, 0, 1]],
                [[ 0, 1, 2,-1], [ 0, 2, 1, 1]],
                [[ 0, 0, 3,-1], [ 0, 3, 0, 1]],
                [[ 0, 1, 3,-1], [ 0, 3, 1, 1]],
                [[ 0, 2, 3,-1], [ 0, 3, 2, 1]]],
        self.assertTrue(numpy.allclose(idx, idx0))

    def test_tn_strs(self):
        self.assertEqual(t1strs(7, 3), cistring.tn_strs(7, 3, 1).tolist())
        self.assertEqual(t2strs(7, 3), cistring.tn_strs(7, 3, 2).tolist())

def t1strs(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    t1s = []
    for a in range(nocc, norb):
        for i in reversed(range(nocc)):
            t1s.append(hf_str ^ (1 << i) | (1 << a))
    return t1s

def t2strs(norb, nelec):
    nocc = nelec
    nvir = norb - nocc
    hf_str = int('1'*nocc, 2)
    ij_allow = [(1<<i)^(1<<j) for i in reversed(range(nocc)) for j in reversed(range(i))]
    ab_allow = [(1<<i)^(1<<j) for i in range(nocc,norb) for j in range(nocc,i)]
    t2s = []
    for ab in ab_allow:
        for ij in ij_allow:
            t2s.append(hf_str ^ ij | ab)
    return t2s

if __name__ == "__main__":
    print("Full Tests for CI string")
    unittest.main()

