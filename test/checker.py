#!/usr/bin/env python3
import sys

argv      = sys.argv[1:]
f7_path        = argv[0]
# plot_path      = argv[1]
# nssd_plot_path = argv[2]

print( f'F7 path: {f7_path}')

f7=40008
f7=f'{f7:-08x}'

challenge=f'{f7}0000000000000000000000000000000000000000000000000000000000000000'[:64]
print( challenge )
