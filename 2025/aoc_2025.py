# Advent of Code, 2025, Sergey Miloykov (sergei.m@gmail.com)

import os
import random
import math
import time
from functools import cmp_to_key
from inspect import currentframe
    


solve_problems = "*"

print('Advent of Code / 2025 / Sergey Miloykov\n')

debug_log = 0

def filter_task(name):
    return name in solve_problems or solve_problems == '*'

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno # <--- Line of caller
    
def get_seconds():
    return time.perf_counter() * 1000.0 # / (10 ** 6)

def get_elapsed_time_str(t0):
    return '{:.2f}'.format((get_seconds() - t0)*1.0) + ' ms'
    
def print_result(text, t0, lines):
    if len(text) < 24:
        text += ' '*(24-len(text))
    text += get_elapsed_time_str(t0)
    if len(text) < 40:
        text += ' '*(40 - len(text))
    text += str(lines+1) + ' loc'
    print(text)


    
global_t0, global_l0 = get_seconds(), get_linenumber()



if filter_task("1a"):
    t0, l0 = get_seconds(), get_linenumber()
    with open('1.txt', 'rt') as fp:
        curr = 50
        code = 0
        for row in fp:
            prev = curr
            number = int(row[1:]) 
            curr += (row[0]=='L') and number or -number
            curr = curr + 99*100000000
            curr = curr % 100
            if curr == 0:
                code += 1
        print_result('1a = ' + str(code), t0, get_linenumber() - l0)

if filter_task("1b"):
    t0, l0 = get_seconds(), get_linenumber()
    with open('1.txt', 'rt') as fp:
        curr = 50
        zeroes = 0
        for row in fp:
            row = row.strip('\n')
            prev = curr
            number = int(row[1:])
            add_number = (row[0]=='L') and number or -number
            curr += add_number
            
            add_click_text = ''
            
            if curr > 0:
                add_clicks = int(math.floor(curr / 100))
                add_click_text = add_clicks > 0 and ' +'+str(add_clicks) or ''
                zeroes += add_clicks
                curr = curr % 100
                
            elif curr < 0:
                add_clicks = int(math.floor(-curr / 100))
                
                if prev > 0:
                    add_clicks += 1
                
                zeroes += add_clicks
                add_click_text = add_clicks > 0 and ' +'+str(add_clicks) or ''                
                curr = curr + 10000000
                curr = curr % 100
            else:
                add_click_text = ' +1'
                zeroes += 1

        print_result('1b = ' + str(zeroes), t0, get_linenumber() - l0)
        
if filter_task("2a"):
    t0, l0 = get_seconds(), get_linenumber()
    with open('2.txt', 'rt') as fp:
        result = 0
        ranges = fp.readline().split(',')        
        for r in ranges:
            r = r.split('-')
            
            range_min_str = r[0]
            range_max_str = r[1]
            
            range_min = int(range_min_str)
            range_max = int(range_max_str)
            
            #print(range_min_str + ' - ' + range_max_str)
            
            found = set()
        
            def iterate_same_digit_numbers(length, a, b):
                summ = 0
                if length % 2 == 0:
                    for digit in range(1,10):
                        x = 0
                        for i in range(length):
                            x = x*10 + digit
                        if x >= a and x <= b and not x in found:
                            #print(x)
                            summ += x
                            found.add(x)
                return summ
                    
            result += iterate_same_digit_numbers(len(range_min_str), range_min, range_max)
            if len(range_min_str) != len(range_max_str):
                result += iterate_same_digit_numbers(len(range_max_str), range_min, range_max)
                
            def iterate_same_halfs_numbers(min_str, max_str):
                summ = 0
                length = len(min_str)
                if length > 3 and length % 2 == 0:
                
                    half_len = int(length/2)
                
                    min_half = min_str[0 : half_len]
                    max_half = max_str[0 : half_len]
                    
                    #print('half ranges - ' + min_half + ', ' + max_half)
                    
                    half_mul = 1
                    for i in range(half_len):
                        half_mul *= 10
                        
                    min_range = int(min_str)
                    max_range = int(max_str)
                
                    for i in range(int(min_half), int(max_half)+1):
                        x = i*half_mul + i
                        if min_range <= x and x <= max_range and not x in found:
                            #print(x)
                            summ += x
                            found.add(x)
                return summ

            if len(range_min_str) == len(range_max_str):
                result += iterate_same_halfs_numbers(range_min_str, range_max_str)
            else:
                result += iterate_same_halfs_numbers(range_min_str, '9' * len(range_min_str))
                result += iterate_same_halfs_numbers('1' + '0'*(len(range_max_str)-1), range_max_str)
                        
        print_result("2a = " + str(result), t0, get_linenumber() - l0)

if filter_task("2b"):
    t0, l0 = get_seconds(), get_linenumber()
    with open('2.txt', 'rt') as fp:
        result = 0
        ranges = fp.readline().split(',')        
        for r in ranges:
            r = r.split('-')
            
            range_min_str = r[0]
            range_max_str = r[1]
            
            range_min = int(range_min_str)
            range_max = int(range_max_str)
            
            #print(range_min_str + ' - ' + range_max_str)
            
            found = set()
        
            def iterate_same_digit_numbers(length, a, b):
                summ = 0
                if length > 1:
                    for digit in range(1,10):
                        x = 0
                        for i in range(length):
                            x = x*10 + digit
                        if x >= a and x <= b and not x in found:
                            #print(x)
                            summ += x
                            found.add(x)
                return summ

            result += iterate_same_digit_numbers(len(range_min_str), range_min, range_max)
            if len(range_min_str) != len(range_max_str):
                result += iterate_same_digit_numbers(len(range_max_str), range_min, range_max)
                
            def iterate_same_halfs_numbers(min_str, max_str):
                summ = 0
                length = len(min_str)
                half_length = int(length/2)
                
                for L in range(2,half_length+1):
                    if length % L > 0:
                        continue
                
                    min_substr = min_str[0 : L]
                    max_substr = max_str[0 : L]
                    
                    #print('half ranges - ' + min_half + ', ' + max_half)
                    
                    min_range = int(min_str)
                    max_range = int(max_str)
                
                    for i in range(int(min_substr), int(max_substr)+1):
                        s = str(i) * int(length/L)
                        x = int(s)
                        
                        if min_range <= x and x <= max_range and not x in found:
                            #print(x)
                            summ += x
                            found.add(x)
                return summ

            if len(range_min_str) == len(range_max_str):
                result += iterate_same_halfs_numbers(range_min_str, range_max_str)
            else:
                result += iterate_same_halfs_numbers(range_min_str, '9' * len(range_min_str))
                result += iterate_same_halfs_numbers('1' + '0'*(len(range_max_str)-1), range_max_str)

        print_result("2b = " + str(result), t0, get_linenumber() - l0)

if filter_task('3a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('3.txt', 'rt') as fp:
        summ = 0
        for row in fp:
            best_battery = 0
        
            row = row.strip('\n')
            
            N = list()
            for x in row:
                N.append(int(x))

            index = [-1]*10            
            pos = 0
            
            for x in N:
                if index[x] == -1:
                    index[x] = pos
                pos = pos + 1
            
            lN = len(N)
            for i in range(9,0,-1):
                first = index[i]
                if first == -1:
                    continue

                best = -1
                for j in range(first+1,lN):
                    if N[j] > best:
                        best = N[j]
                
                if best >= 0:
                    best_battery = i*10 + best
                    break
            
            summ += best_battery
            
    print_result("3a = " + str(summ), t0, get_linenumber() - l0)

if filter_task('3b'):
    t0, l0 = get_seconds(), get_linenumber()
    
    def find_best(M, batteries, pos, length):
        L = len(batteries)
    
        if length == 0:
            return 0
            
        if pos+length > L:
            return -1
    
        if M[pos][length] > -1:
            return M[pos][length]
            
        mul10 = 1
        for i in range(length-1):
            mul10 *= 10

        # try iterating from largest starting digits to the smallest - so when we get a successful sequence we can skip searching with smaller starting digits
        sorted_range = []
        for i in range(pos, L):
            sorted_range.append(i)
        sorted_range.sort(key = lambda item : batteries[item], reverse = True)

        best = -1
        best_first_digit = 0
        
        #for i in range(pos,L):
        for i in sorted_range:
            if batteries[i] >= best_first_digit:
                rec = find_best(M, batteries, i+1, length-1)
                if rec >= 0:
                    x = batteries[i]*mul10 + rec
                    if x > best:
                        best = x
                        best_first_digit = batteries[i]
                    
        M[pos][length] = best
        return best

    with open('3.txt', 'rt') as fp:
        summ = 0
        for row in fp:
            row = row.strip('\n')
            
            batteries = list()
            for x in row:
                batteries.append(int(x))
            
            M = []
            for i in range(len(batteries)):
                M.append( [-1]*13 )

            best = find_best(M, batteries, 0, 12)
            summ += best
            
        print_result("3b = " + str(summ), t0, get_linenumber() - l0)
        
if filter_task('4a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('4.txt', 'rt') as file:
        grid = []
        for row in file:
            row = row.strip('\n')
            grid_line = []
            for r in row:
                grid_line.append(r)
            grid.append(grid_line)
        
        h = len(grid)
        w = len(grid[0])
        dirs = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1]]

        result = 0
        for y in range(0,h):
            for x in range(0,w):
                rolls = 0
                if grid[y][x] != '@':
                    continue
                    
                if x == 0 or y == 0 or x+1 == w or y+1 == h:
                    for d in dirs:
                        nx = x + d[0]
                        ny = y + d[1]
                        if nx>=0 and ny>=0 and nx<w and ny<h:
                            if grid[ny][nx] == '@':
                                rolls += 1
                                if rolls == 4:
                                    break
                else:
                    for d in dirs:
                        nx = x + d[0]
                        ny = y + d[1]
                        if grid[ny][nx] == '@':
                            rolls += 1
                            if rolls == 4:
                                break
                
                if rolls < 4:
                    result += 1
                    
        print_result('4a = ' + str(result), t0, get_linenumber() - l0)

if filter_task('4b'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('4.txt', 'rt') as file:
        grid = []
        for row in file:
            row = row.strip('\n')
            grid_line = []
            for r in row:
                grid_line.append(r)
            grid.append(grid_line)
        
        h = len(grid)
        w = len(grid[0])
        dirs = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1]]

        result = 0

        to_remove = []
        to_iterate = set() # remember rolls around the one we consider for removal, so we can iterate only them in the next pass
        
        def iterate(x, y):
            if grid[y][x] != '@':
                return 0
                
            toi = []
            rolls = 0
            
            if x==0 or y==0 or x+1==w or y+1==h:
                for d in dirs:
                    nx = x + d[0]
                    ny = y + d[1]
                
                    if nx>=0 and ny>=0 and nx<w and ny<h:
                        if grid[ny][nx] == '@':
                            rolls += 1
                            toi.append(nx + ny*w)
                            if rolls == 4:
                                return 0
            else:
                for d in dirs:
                    nx = x + d[0]
                    ny = y + d[1]
                
                    if grid[ny][nx] == '@':
                        rolls += 1
                        toi.append(nx + ny*w)
                        if rolls == 4:
                            return 0
            
            to_remove.append([x,y])
            for x in toi:
                to_iterate.add(x)
                        
            return 1

        # iterate over the whole grid only the first time
        
        for y in range(h):
            for x in range(w):
                if grid[y][x] == '@':
                    result += iterate(x,y)

        if len(to_remove) > 0:
            for coord in to_remove:
                x,y = coord
                grid[y][x] = '.'

        # iterate over the active set populated by the neighbours of the deleted rolls
        while len(to_iterate) > 0:
            curr_iteration = to_iterate
            to_iterate = set()
            
            for coord in curr_iteration:
                x = coord % w
                y = int(coord / w)
                result += iterate(x, y)
                
            if len(to_remove) == 0:
                break
                
            for coord in to_remove:
                grid[coord[1]][coord[0]] = '.'
            to_remove = []

        print_result('4b = ' + str(result), t0, get_linenumber() - l0)
            
if filter_task('5a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('5.txt', 'rt') as fp:
        ranges = []
        
        for row in fp:
            row = row.strip('\n')
            if '-' in row:
                r = row.split('-')
                a = r[0]
                b = r[1]
                ranges.append([ int(a) + 0.0, 0])
                ranges.append([ int(b) + 0.2, 2])
            elif row != '':
                ranges.append([ int(row) + 0.1, 1])
        
        # sort ranges and numbers, so if a number or range overlap - start of a range comes first, number 2nd and end of range 3rd - that way numbers will fall inside ranges correctly
        #ranges = sorted(ranges, key = cmp_to_key(lambda item1, item2: item1[0]-item2[0] if item1[0] != item2[0] else item1[1] - item2[1]))
        ranges.sort(key = lambda item: item[0])

        inside = 0
        result = 0
        for r in ranges:
            v = r[1]
            if v == 0:
                inside += 1
            elif v == 2:
                inside -= 1
            elif inside>0:
                result += 1
        
        print_result('5a = ' + str(result), t0, get_linenumber() - l0)
        
if filter_task('5b'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('5.txt', 'rt') as fp:
        ranges = []
        result = 0
        for row in fp:
            row = row.strip('\n')
            if '-' in row:
                r = row.split('-')
                ranges.append([int(r[0]), True])
                ranges.append([int(r[1]), False])

        ranges.sort(key = lambda item: item[0])
        
        inside = 1
        last = ranges[0][0]
        last_inside = 1
        last_extra = False
        
        for i in range(1,len(ranges)):
            x = ranges[i]
            
            if x[1]:
                inside += 1
                if last_extra and last == x[0]:
                    result -= 1
            else:
                inside -= 1
            
            last_extra = False
            if inside >= 0 and last_inside > 0:
                result += x[0] - last
                
                if inside == 0:
                    last_extra = True
                    result += 1
                
            last = x[0]
            last_inside = inside
        
        print_result('5b = ' + str(result), t0, get_linenumber() - l0)

if filter_task('6a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('6.txt', 'rt') as fp:
        numbers = []
        for row in fp:
            if row[0] == '+' or row[0] == '*':
                ops = row.strip('\n').split()
            else:
                numbers.append(row.strip('\n').split())
                
        result = 0
        for i in range(0,len(ops)):
            op = ops[i]
            acc = int((op == '+') and '0' or '1')
            for j in range(0,len(numbers)):
                if op == '+':
                    acc += int(numbers[j][i])
                else:
                    acc *= int(numbers[j][i])
            result += acc
        print_result('6a = ' + str(result), t0, get_linenumber() - l0)
            
if filter_task('6b'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('6.txt', 'rt') as fp:
        numbers = []
        for row in fp:
            if row[0] == '+' or row[0] == '*':
                ops = row.strip('\n')
            else:
                numbers.append(row.strip('\n'))
        
        L = len(ops)
        for n in numbers:
            if len(n) > L:
                L = len(n)
        while len(ops) < L:
            ops += ' '
        for n in numbers:
            while len(n) < L:
                n += ' '
                
        result = 0

        N = len(numbers)
        for i in range(L):
            if ops[i] == ' ':
                continue
            op = ops[i]
            acc = int((op == '+') and '0' or '1')
            
            j = i
            while j < L and (j == i or ops[j] == ' '):
                number = 0
                for k in range(N):
                    if numbers[k][j] != ' ':
                        number = number*10 + int(numbers[k][j])

                if op == '+':
                    acc += number
                elif number > 0:
                    acc *= number
                j += 1
            
            result += acc
        print_result('6b = ' + str(result), t0, get_linenumber() - l0)

if filter_task('7a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('7.txt', 'rt') as fp:
        splits = 0
        w = 0
        active = set()
        for row in fp:
            row = row.strip('\n')
            
            if len(active) == 0:
                x = row.find('S')
                w = len(row)
                active.add(x)

            next_active = set()
            for i in active:
                if row[i] == '^':
                    if i-1>=0:
                        next_active.add(i-1)
                    if i+1<w:
                        next_active.add(i+1)
                    splits += 1
                else:
                    next_active.add(i)
            active = next_active
    
        print_result('7а = ' + str(splits), t0, get_linenumber() - l0)
        
if filter_task('7b'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('7.txt', 'rt') as fp:
        w = 0
        wave = dict()
        next_wave = dict()

        def add_particle(wave, position, count):
            if not position in wave:
                wave[position] = count
            else:
                wave[position] += count

        for row in fp:
            row = row.strip('\n')
            
            if len(wave) == 0:
                x = row.find('S')
                w = len(row)
                wave[x] = 1

            next_wave = dict()
            for pos,count in wave.items():
                if row[pos] == '^':
                    if pos-1 >= 0:
                        add_particle(next_wave, pos-1, count)
                    if pos+1 < w:
                        add_particle(next_wave, pos+1, count)
                else:
                    add_particle(next_wave, pos, count)
            wave = next_wave
        
        particles = 0
        for pos,count in wave.items():
            particles += count
    
        print_result('7b = ' + str(particles), t0, get_linenumber() - l0)

boxes8ab = []
distances8ab = []

def load_boxes_and_compute_distance_matrix():
    with open('8.txt', 'rt') as fp:
        for row in fp:
            row = row.strip('\n')
            coords = row.split(',')
            boxes8ab.append([int(coords[0]), int(coords[1]), int(coords[2])])
        N = len(boxes8ab)
        for i in range(N):
            b1 = boxes8ab[i]
            for j in range(i+1,N):
                b2 = boxes8ab[j]
                dx = b1[0] - b2[0]
                dy = b1[1] - b2[1]
                dz = b1[2] - b2[2]
                d = dx*dx + dy*dy + dz*dz
                distances8ab.append([d, i, j])
        distances8ab.sort(key = lambda item: item[0])
        return N

if filter_task('8a'):
    t0, l0 = get_seconds(), get_linenumber()

    load_boxes_and_compute_distance_matrix()
    N = len(boxes8ab)
    
    graph = []
    for i in range(N):
        graph.append([])
        
    def add_edge(i,j):
        graph[i].append(j)
        graph[j].append(i)
        
    for i in range(0,1000):
        edge = distances8ab[i]
        add_edge(edge[1], edge[2])
    
    used = set()
    
    def walk_subgraph(i):
        if i in used:
            return 0
        size = 1
        used.add(i)
        for x in graph[i]:
            size += walk_subgraph(x)
        return size
    
    subgraph_sizes = []
    for i in range(N):
        size = walk_subgraph(i)
        if size > 1:
            subgraph_sizes.append(size)
    subgraph_sizes.sort(reverse = True)
    result = subgraph_sizes[0] * subgraph_sizes[1] * subgraph_sizes[2]
    
    print_result('8a = ' + str(result), t0, get_linenumber() - l0)
        
if filter_task('8b'):
    t0, l0 = get_seconds(), get_linenumber()
    
    if len(distances8ab) == 0:
        load_boxes_and_compute_distance_matrix()
    N = len(boxes8ab)
        
    graph = []
    for i in range(N):
        graph.append([])
        
    subgraph_ids = dict() # box -> subgraph ID
    subgraphs = dict()    # subgraph ID -> set of box indices
        
    def add_edge(i,j):
        graph[i].append(j)
        graph[j].append(i)
        
        subi = i in subgraph_ids and subgraph_ids[i] or -1
        subj = j in subgraph_ids and subgraph_ids[j] or -1            

        if subi == -1 and subj == -1:
            subgraph_ids[i] = i
            subgraph_ids[j] = i
            subgraphs[i] = set()
            subgraphs[i].add(i)
            subgraphs[i].add(j)
        elif subi > -1 and subj == -1:
            subgraph_ids[j] = subi
            subgraphs[subi].add(j)
        elif subi == -1 and subj > -1:
            subgraph_ids[i] = subj
            subgraphs[subj].add(i)
        elif subi != subj:
            for x in subgraphs[subj]:
                subgraph_ids[x] = subi
                subgraphs[subi].add(x)
            if len(subgraphs[subi]) == N:
                return boxes8ab[j][0] * boxes8ab[i][0]
        return 0

    result = 0
    for edge in distances8ab:
        #print(str(i) + ' - ' + str(edge[0]))
        result = add_edge(edge[1], edge[2])
        if result > 0:
            break
    
    print_result('8b = ' + str(result), t0, get_linenumber() - l0)

if filter_task('9a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('9.txt', 'rt') as fp:
        pts = []
        for row in fp:
            row = row.strip('\n')
            if row != '':
                row = row.split(',')
                pts.append([int(row[0]), int(row[1])])

        best_area = 0
        N = len(pts)
        for i in range(N-1):
            for j in range(i+1,N):
                area = (abs(pts[i][0] - pts[j][0])+1) * (abs(pts[i][1] - pts[j][1])+1)
                if area > best_area:
                    best_area = area
        print_result('9a = ' + str(best_area), t0, get_linenumber() - l0)

if filter_task('9b'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('9.txt', 'rt') as fp:
    
        pts = []
        for row in fp:
            row = row.strip('\n')
            if row != '':
                row = row.split(',')
                x = int(row[0])
                y = int(row[1])
                pts.append([x, y])

        N = len(pts)
        
        bx = pts[0][0]
        Bx = bx

        for i in range(1,N):
            x,y = pts[i]
            if x < bx:
                bx = x
            if x > Bx:
                Bx = x
                
        bound_size = Bx - bx
        
        prev = N-1
        prev_dist = 0
        
        corners = []
        for i in range(N):
            dx = pts[i][0] - pts[prev][0]
            dy = pts[i][1] - pts[prev][1]
            d = abs(dx) + abs(dy)
            
            if d > bound_size/2:
                corners.append(len(corners)>0 and prev or i)
            prev = i
                
        def intersect_rect(mx,my,Mx,My,i,j):
            xi = pts[i][0]
            yi = pts[i][1]
            xj = pts[j][0]
            yj = pts[j][1]
            
            if xi == xj:
                if xi <= mx or xi >= Mx:
                    return False
                    
                if yi < yj:
                    y0,y1 = yi,yj
                else:
                    y0,y1 = yj,yi
                    
                return y0 < My and y1 > my
            else:
                if yi <= my or yi >= My:
                    return False
                    
                if xi < xj:
                    x0,x1 = xi, xj
                else:
                    x0,x1 = xj, xi
                    
                return x0 < Mx and x1 > mx
            
        
        best_area = 0
        
        for i in corners:
            xi,yi = pts[i]
            for j in range(N):
                if j == i:
                    continue
                xj,yj = pts[j]
                area = (abs(xi-xj)+1) * (abs(yi-yj)+1)
                if area > best_area:
                    mx,my = xi,yi
                    Mx,My = xj,yj
                    if mx > Mx:
                        mx,Mx = Mx,mx
                    if my > My:
                        my,My = My,my
                       
                    prev = N-1
                    intersection = False
                    for k in range(N):
                        if intersect_rect(mx,my,Mx,My,k,prev):
                            intersection = True
                            break
                        prev = k
                            
                    if not intersection:
                        best_area = area

        print_result('9b = ' + str(best_area), t0, get_linenumber() - l0)

if filter_task('10a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('10.txt', 'rt') as fp:
        result = 0
        for row in fp:
            row = row.strip('\n')
            if row == '':
                continue
                
            scheme = row.split(' ')

            mask = 0
            bit = 1
            for x in scheme[0][1:-1]:
                if x == '#':
                    mask = mask | bit
                bit = bit * 2
                
            #print('Mask = ' + str(scheme[0]) + ' -> ' + str(mask))
            
            buttons = []
            loops = 1
            
            
            for i in range(1,len(scheme)-1):
                b = scheme[i][1:-1].split(',')
                
                btn = 0
                for x in b:
                    btn |= 1 << int(x)
                buttons.append(btn)
                loops *= 2
                
                #print(str(b) + ' -> ' + str(btn))
            best_press = 1000000
            result = 0
            for i in range(1,loops):
                
                test = 0
                bmask = 1
                press = 0
                for j in range(len(buttons)):
                    if i & bmask:
                        test ^= buttons[j]
                        press += 1
                    bmask *= 2
                    
                if test == mask and press < best_press:
                    best_press = press
            #print("Found " + str(best_press) + " press number")
            result += best_press
                    
        print_result('10a = ' + str(result), t0, get_linenumber() - l0)

if filter_task('10b'):
    t0, l0 = get_seconds(), get_linenumber()
    
    def simplify_integer_pair(a,b):
        if abs(a) == abs(b):
            return a//abs(a), b//abs(b)
            
        if a % abs(b) == 0:
            return a // abs(b), b // abs(b)
        
        if b % abs(a) == 0:
            return b // abs(a), a // abs(a)
            
        return a,b
        
    def iterate_loops(loops, curr): # iterate once several nested loops [[inner_curr_value, inner_max_value], ..., [outer_curr_value, outer_max_value]]
        loops[curr][0] += 1
        if loops[curr][0] == loops[curr][1]:
            if curr+1 == len(loops):
                return -1
            for i in range(curr+1):
                loops[i][0] = 0
            curr += 1
            
            return iterate_loops(loops, curr)
        else:
            curr = 0
        return curr

    best_solution = False
    best_solution_sum = -1

    def solve_gaussian_triangle_row(eq, current_row, solved_vars = [], solved_vars_sum = 0, solved_vars_count = 0, ident = ''):
        global best_solution
        global best_solution_sum


        N = len(eq)
        K = len(eq[0])
        
        if debug_log > 1:
            if current_row >= 0:
                print(ident + "sgtr(" + str(current_row) + "): " + str(eq[current_row]) + " solved_num: " + str(solved_vars_count) + " solved_sum: " + str(solved_vars_sum) + ' solved_vars = ' + str(solved_vars))
            else:
                print(ident + "sgtr(-1)")
            ident += '  '
        
        if solved_vars_count == K-1:
            if best_solution_sum == -1 or solved_vars_sum < best_solution_sum:
                best_solution_sum = solved_vars_sum
                best_solution = solved_vars.copy()
                #print("solution FOUND: " + str(best_solution) + " sum = " + str(best_solution_sum) + " count = " + str(solved_vars_count))
            return True
            
        if current_row < 0:
            return False
        
        if len(solved_vars) == 0:
            solved_vars = [-1]*(K-1)
            
        unsolved_vars = []
        summ = 0    
        for i in range(K-1): # accumulate solved variables and gather unsolved ones
            if eq[current_row][i] != 0:
                if solved_vars[i] >= 0:
                    summ += eq[current_row][i] * solved_vars[i]
                else:
                    unsolved_vars.append(i)
                    
        B = eq[current_row][-1] - summ
        
        if len(unsolved_vars) == 0:
            if B == 0: # we 'know' all the vars for that equation and it isn't correct - fail and go back
                solve_gaussian_triangle_row(eq, current_row-1, solved_vars, solved_vars_sum, solved_vars_count, ident)        
            return False

        if len(unsolved_vars) > 1:
            loop = []
            negative_coefs = False
            for x in unsolved_vars:
                A = eq[current_row][x]
                loop.append([0, abs(B//A)+1])
                if A < 0:
                    negative_coefs = True
                #if B==0:
            
            if negative_coefs:
                for x in loop:
                    x[1] = max( 35, x[1])
                
            loop.pop(-1) # remove one variable from looping, as we will compute it from the other ones anyway
            
            
            A = eq[current_row][unsolved_vars[-1]]  # pick last unsolved var, that won't be looped over to compute it directly when looping
            
            if debug_log > 1:
                print(ident + "More than one unsolved parameter - looping through all the variants, deducing variable " + str(unsolved_vars[-1]) + " with coef " + str(A))
            
            loop_i = 0
            while loop_i >= 0:
            
                if debug_log > 1:
                    print(ident + str(loop))
            
                loop_sum = 0
                for i in range(len(loop)):
                    loop_sum += eq[current_row][unsolved_vars[i]] * loop[i][0]
                    
                loop_B = B - loop_sum
                if loop_B % A == 0 and loop_B // A >= 0:
                    last_row_var = loop_B // A
                    
                    #print(ident + "B = " + str(B) + " loop_B = " + str(loop_B) + " last_row_var = " + str(last_row_var) + " loop_sum = " + str(loop_sum) + " unsolved = " + str(unsolved_vars))

                    loop_solved_sum = 0
                    for i in range(len(loop)):
                        solved_vars[unsolved_vars[i]] = loop[i][0]
                        loop_solved_sum += loop[i][0]
                    solved_vars[unsolved_vars[-1]] = last_row_var
                    loop_solved_sum += last_row_var
                    
                    if best_solution_sum == -1 or solved_vars_sum + loop_solved_sum < best_solution_sum:
                        solve_gaussian_triangle_row(eq, current_row - 1, solved_vars, solved_vars_sum + loop_solved_sum, solved_vars_count + len(unsolved_vars), ident)
                    
                    for i in range(len(unsolved_vars)):
                        solved_vars[unsolved_vars[i]] = -1
                else:
                    #print(ident + "B = " + str(B) + " loop_B = " + str(loop_B) + " NEGATIVE last_row_var = " + str(loop_B // A) + " loop_sum = " + str(loop_sum) + " unsolved = " + str(unsolved_vars))
                    jadjs = 1
                    
                loop_i = iterate_loops(loop, loop_i)

            return False
        
        if len(unsolved_vars) == 1: # solve Ax = B for the one unknown parameter
        
            A = eq[current_row][ unsolved_vars[0] ]
            
            if B % A > 0:
                if debug_log > 1:
                    print(ident + "Exactly one solution - failed!")
                return False
                
            solution = B // A
            
            if solution >= 0:
                if debug_log > 1:
                    print(ident + "solved parameter " + str(unsolved_vars[0]) + " = " + str(solution))
                    
                if best_solution_sum != -1 and solved_vars_sum + solution > best_solution_sum:
                    return False
                    
                solved_vars[unsolved_vars[0]] = solution
                solve_gaussian_triangle_row(eq, current_row-1, solved_vars, solved_vars_sum + solution, solved_vars_count + 1, ident)
                solved_vars[unsolved_vars[0]] = -1
        
        return True
        
    def solve_equations(eq, task_index):
        global best_solution
        global best_solution_sum
        
        N = len(eq)
        K = len(eq[0]) - 1
        iterations = min(N, K)
        
        # gaussian reduction
        for i in range(iterations):
            for j in range(i,N):
                if eq[j][i] < 0:
                    for k in range(len(eq[j])):
                        eq[j][k] = -eq[j][k]

            eq[i:N] = sorted(eq[i:N], reverse = True, key = lambda item: item[i])
            
            for j in range(i+1,N):
                if eq[j][i] != 0:
                    a, b = simplify_integer_pair(eq[i][i], eq[j][i])
                    if debug_log > 1:
                        print(str(eq[j]) + ' * ' + str(a) + ' - ' + str(eq[i]) + ' * ' + str(b))
                    for k in range(len(eq[j])):
                        eq[j][k] = eq[j][k]*a - eq[i][k]*b

            if debug_log > 1:
                print("Guassian reduction step " + str(i))
                for e in eq:
                    print('  ' + str(e))
                    
        while True:
            zero = True
            for x in eq[-1]:
                if x != 0:
                    zero = False
                    break
            if zero:
                eq.pop(-1)
            else:
                break
                
        if debug_log > 1:
            print("Optimization")
            for e in eq:
                print('  ' + str(e))
        
        if debug_log == 1:
            print("Guassian reduction:")
            for e in eq:
                print('  ' + str(e))
        
        best_solution = []
        best_solution_sum = -1
        
        solve_gaussian_triangle_row(eq, len(eq)-1)
        
        if debug_log > 0:
            if best_solution_sum != -1:
                print("Solution " + str(task_index) + " = " + str(best_solution_sum) + '  ' + str(best_solution))
            else:
                print("FAILED to find solution!")
                
        return best_solution_sum
    
    with open('10.txt', 'rt') as fp:
        result = 0

        task_index = 1
        solve_only_task = False        
        
        for row in fp:
            if solve_only_task and solve_only_task != task_index:
                debug_log = 1
                task_index += 1
                continue
        
            row = row.strip('\n')
            if row != '':
                scheme = row.split(' ')
                
                target = scheme[-1][1:-1].split(',')
                for i in range(len(target)):
                    target[i] = int(target[i])
                    
                scheme.pop(0)
                scheme.pop(-1)
                
                if debug_log > 0:
                    print("Buttons size = " + str(len(scheme)-1))
                    print("Target size  = " + str(len(target)))
                
                buttons = []
                    
                for s in scheme:
                    b = s[1:-1].split(',')
                    for i in range(len(b)):
                        b[i] = int(b[i])
                        
                    buttons.append(b)
                    if debug_log > 0:
                        print(b)
                
                eq = []
                for i in range(len(target)):
                    eq.append([0] * (len(buttons) + 1))
                    eq[-1][-1] = target[i]
                    
                for i in range(len(buttons)):
                    for x in buttons[i]:
                        eq[x][i] = 1
                
                if debug_log > 0:
                    print('Equation:')
                    for e in eq:
                        print('  ' + str(e))
                        
                result += solve_equations(eq, task_index)
                task_index += 1
                
                #exit()
                
                if debug_log > 0:
                    print('\n\n')

        print_result('10b = ' + str(result), t0, get_linenumber() - l0)

if filter_task('11a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('11.txt', 'rt') as fp:
        graph = dict()
        for row in fp:
            row = row.strip('\n')
            row = row.split(':')
            head = row[0]
            other = row[1].split(' ')
            
            graph[head] = []
            for x in other:
                if x != '':
                    graph[head].append(x)
        result = 0
        wave = ['you']
        
        while len(wave) > 0:
            next_wave = []
            for x in wave:
                for n in graph[x]:
                    if n == 'out':
                        result += 1
                    else:
                        next_wave.append(n)
            wave = next_wave
            
        print_result('11a = ' + str(result), t0, get_linenumber() - l0)
            
if filter_task('11b'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('11.txt', 'rt') as fp:
        graph = dict()
        for row in fp:
            row = row.strip('\n')
            row = row.split(':')
            head = row[0]
            other = row[1].split(' ')
            graph[head] = []
            for x in other:
                if x != '':
                    graph[head].append(x)
        graph["out"] = []
            
        result = 0
        def find_path_count(a,b,excludes):
            #print('finding a path from ' + a + ' to ' + b)
            
            ws = set()
            ws.add(a)
            for e in excludes:
                ws.add(e)
            
            wave = [a]
            count = 0
            
            counts = dict()
            counts[b] = 1
            for e in excludes:
                counts[e] = 0
            
            def find_count(x):
                if x in counts:
                    return counts[x]
                    
                summ = 0
                for n in graph[x]:
                    summ += find_count(n)
                
                counts[x] = summ
                return summ
                
            return find_count(a)
            
        svr2dac = find_path_count('svr', 'dac', ['fft', 'out'])
        dac2fft = find_path_count('dac', 'fft', ['svr', 'out'])
        fft2out = find_path_count('fft', 'out', ['svr', 'dac'])
        
        svr2fft = find_path_count('svr', 'fft', ['dac', 'out'])
        fft2dac = find_path_count('fft', 'dac', ['svr', 'out'])
        dac2out = find_path_count('dac', 'out', ['fft', 'svr'])
        
        result = svr2fft * fft2dac * dac2out + svr2dac * dac2fft * fft2out
        print_result('11b = ' + str(result), t0, get_linenumber() - l0)

if filter_task('12a'):
    t0, l0 = get_seconds(), get_linenumber()
    with open('12.txt', 'rt') as fp:
        area = []
        result = 0
        for row in fp:
            row = row.strip('\n')
            if 'x' in row:
                ab = row.split(':')
                size = ab[0].split('x')
                
                counts = ab[1].split(' ')
                if counts[0] == '':
                    counts.pop(0)
                    
                w = int(size[0])
                h = int(size[1])
                
                m = 0
                M = 0
                for i in range(len(counts)):
                    counts[i] = int(counts[i])
                    m += counts[i]*area[i]
                    M += counts[i]*9
                
                w3 = w - w % 3
                h3 = h - h % 3
                
                if w3*h3 >= M:
                    result += 1

                if False:
                    easy_test = ''
                    if w*h < m:
                        easy_test = "impossible"
                    elif w3*h3 >= M:
                        easy_test = "TRIVIAL"
                        result += 1
                    print(str(w) + 'x' + str(h) + ': ' + str(w*h) + ' ' + str(w3*h3) + ' ' + str(m) + ' - ' + str(M) + '  ' + easy_test)
                
            elif '.' in row or '#' in row:
                row = row.replace('.', '')
                area[-1] += len(row)
            elif ':' in row:
                area.append(0)
        
        print_result('12a = ' + str(result), t0, get_linenumber() - l0)
            

print("\nTotal time: " + get_elapsed_time_str(global_t0))
print("Total loc: " + str(get_linenumber() - global_l0))
            