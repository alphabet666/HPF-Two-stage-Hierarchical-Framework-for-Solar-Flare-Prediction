import sys
import os
import re
import itertools


class Solution:
    def solution(self, d):
        b = []
        for j in range(len(d)+1):
            ar_iter = itertools.permutations(d, j)
            for num in ar_iter:
                if sum(num) % 3 == 0:
                    a = 0
                    for i in range(j):
                        a += num[i] * (10 ** i)
                        b.append(a)
                else:
                    b.append(0)
        return str(max(b))



# # Write Code Here
#
_d_cnt = 0
_d_cnt = int(input())
_d_i = 0
_d = []
while _d_i < _d_cnt:
    _d_item = int(input())
    _d.append(_d_item)
    _d_i += 1

s = Solution()
res = s.solution(_d)

print(res + "\n")