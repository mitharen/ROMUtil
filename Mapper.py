#!/usr/bin/env python

import sys
import AreaParser
from pyomo.environ import *
import svgwrite
from svgwrite import cm

def main():
    parser = AreaParser.Parser()
    with open(sys.argv[1], 'r') as f:
        area = parser.parse(f.read())

    for section in area:
        if section[0] == '#ROOMS':
            rooms = section[1]
            print('%s: %d'%(section[0], len(section[1]) if section[1] else 0))
        elif section[0] == '#AREA':
            area = section[1]
            print(area)
#    return

    rdb = {}
    # clean up garbage extras
    for room in rooms:
        rdb[room[0]] = (room[1], [] if not room[2] else [e for e in room[2] if e is not None])

    # remove edges to other areas
    for key,room in rdb.items():
        rdb[key] = (room[0], list(filter(lambda e: e[1] in rdb, room[1])))

    # collect exit pairs and one-ways
    exits = []
    for key, room in rdb.items():
        for ex in room[1]:
            if not len(list(filter(lambda e: key == e[1], rdb[ex[1]][1]))):
                exits.append(((key, ex[1]), ex[0], True))
            elif key < ex[1]:
                exits.append(((key, ex[1]), ex[0], False))

    model = solve(rdb, exits)
    
    dwg = svgwrite.Drawing(sys.argv[1]+'.svg', profile='tiny')
    dwg_exits = dwg.add(dwg.g(id='exits', stroke='black'))
    dwg_rooms = dwg.add(dwg.g(id='rooms', fill='red'))

    x_min = min([model.x[i].value for i in model.x if model.x[i].value])
    y_min = min([model.y[i].value for i in model.y if model.x[i].value])

    for ex in exits:
        start, end = ex[0][0], ex[0][1]
        dwg_exits.add(dwg.line(start=((model.x[start].value+.25-x_min)*cm,
                                      (model.y[start].value+.25-y_min)*cm),
                               end=((model.x[end].value+.25-x_min)*cm,
                                    (model.y[end].value+.25-y_min)*cm)))
    for room in rdb.keys():
        if model.x[room].stale == True: continue
        dwg_rooms.add(dwg.rect(insert=((model.x[room].value-x_min)*cm,
                                       (model.y[room].value-y_min)*cm), size=(.5*cm,.5*cm)))
    dwg.save()

def solve(rdb, exits):
    # construct model
    model = ConcreteModel()
    model.Rooms = Set(initialize=rdb.keys())
    model.Exits = RangeSet(0, len(exits)-1)
    model.Directions = RangeSet(0, 6)
    model.M = Param(initialize=len(exits))
    # room position
    model.x = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    model.y = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    model.z = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    # max exit lengths
    model.l = Var(model.Exits, within=PositiveIntegers)
    model.l_min = Param(initialize=1)
    # objective to minimize max exit lengths
    model.obj = Objective(expr=sum([model.l[e] for e in model.Exits]))

    # constraints for exit crossings
    model.crossings = ConstraintList()

    # constraints for relative position of rooms
    model.relations = Var(model.Exits, model.Exits, model.Directions, within=Boolean)
    model.relative_pos = ConstraintList()

    # add constraints
    for i, ex in enumerate(exits):
        print('[*] Exit %d/%d'%(i+1,len(exits)))

        # relative position O(e)
        if ex[1] not in (1, 3):
            model.relative_pos.add(model.x[ex[0][0]] == model.x[ex[0][1]])
        if ex[1] not in (0, 2):
            model.relative_pos.add(model.y[ex[0][0]] == model.y[ex[0][1]])
        if ex[1] not in (4, 5):
            model.relative_pos.add(model.z[ex[0][0]] == model.z[ex[0][1]])

        if ex[1] == 0: # north
            model.relative_pos.add(model.y[ex[0][0]] + model.l_min <= model.y[ex[0][1]])
            model.relative_pos.add(model.y[ex[0][0]] + model.l[i] >= model.y[ex[0][1]])
        elif ex[1] == 1: # east
            model.relative_pos.add(model.x[ex[0][0]] + model.l_min <= model.x[ex[0][1]])
            model.relative_pos.add(model.x[ex[0][0]] + model.l[i] >= model.x[ex[0][1]])
        elif ex[1] == 2: # south
            model.relative_pos.add(model.y[ex[0][0]] >= model.y[ex[0][1]] + model.l_min)
            model.relative_pos.add(model.y[ex[0][0]] <= model.y[ex[0][1]] + model.l[i])
        elif ex[1] == 3: # west
            model.relative_pos.add(model.x[ex[0][0]] >= model.x[ex[0][1]] + model.l_min)
            model.relative_pos.add(model.x[ex[0][0]] <= model.x[ex[0][1]] + model.l[i])
        elif ex[1] == 4: # up
            model.relative_pos.add(model.z[ex[0][0]] + model.l_min <= model.z[ex[0][1]])
            model.relative_pos.add(model.z[ex[0][0]] + model.l[i] >= model.z[ex[0][1]])
        elif ex[1] == 5: # down
            model.relative_pos.add(model.z[ex[0][0]] >= model.z[ex[0][1]] + model.l_min)
            model.relative_pos.add(model.z[ex[0][0]] <= model.z[ex[0][1]] + model.l[i])
        else:
            print('[!] New Direction')

        # ex crossings O(e^2)
        non_incidents = [j for j,v in enumerate(exits) if \
                         ex[0][0] not in v[0] and ex[0][1] not in v[0]]
        for non_incident in non_incidents:
            a, b = i, non_incident
            model.crossings.add(sum([model.relations[a,b,i] for i in range(6)]) >= 1)
            # north
            model.crossings.add(model.y[exits[a][0][0]] - model.y[exits[b][0][0]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,0]))
            model.crossings.add(model.y[exits[a][0][0]] - model.y[exits[b][0][1]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,0]))
            model.crossings.add(model.y[exits[a][0][1]] - model.y[exits[b][0][0]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,0]))
            model.crossings.add(model.y[exits[a][0][1]] - model.y[exits[b][0][1]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,0]))
            # east
            model.crossings.add(model.x[exits[a][0][0]] - model.x[exits[b][0][0]] <= \
                                model.M*(1 - model.relations[a,b,1]) - model.l_min)
            model.crossings.add(model.x[exits[a][0][0]] - model.x[exits[b][0][1]] <= \
                                model.M*(1 - model.relations[a,b,1]) - model.l_min)
            model.crossings.add(model.x[exits[a][0][1]] - model.x[exits[b][0][0]] <= \
                                model.M*(1 - model.relations[a,b,1]) - model.l_min)
            model.crossings.add(model.x[exits[a][0][1]] - model.x[exits[b][0][1]] <= \
                                model.M*(1 - model.relations[a,b,1]) - model.l_min)
            # south
            model.crossings.add(model.y[exits[a][0][0]] - model.y[exits[b][0][0]] <= \
                                model.M*(1 - model.relations[a,b,2]) - model.l_min)
            model.crossings.add(model.y[exits[a][0][0]] - model.y[exits[b][0][1]] <= \
                                model.M*(1 - model.relations[a,b,2]) - model.l_min)
            model.crossings.add(model.y[exits[a][0][1]] - model.y[exits[b][0][0]] <= \
                                model.M*(1 - model.relations[a,b,2]) - model.l_min)
            model.crossings.add(model.y[exits[a][0][1]] - model.y[exits[b][0][1]] <= \
                                model.M*(1 - model.relations[a,b,2]) - model.l_min)
            # west
            model.crossings.add(model.x[exits[a][0][0]] - model.x[exits[b][0][0]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,3]))
            model.crossings.add(model.x[exits[a][0][0]] - model.x[exits[b][0][1]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,3]))
            model.crossings.add(model.x[exits[a][0][1]] - model.x[exits[b][0][0]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,3]))
            model.crossings.add(model.x[exits[a][0][1]] - model.x[exits[b][0][1]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,3]))
            # up
            model.crossings.add(model.z[exits[a][0][0]] - model.z[exits[b][0][0]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,4]))
            model.crossings.add(model.z[exits[a][0][0]] - model.z[exits[b][0][1]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,4]))
            model.crossings.add(model.z[exits[a][0][1]] - model.z[exits[b][0][0]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,4]))
            model.crossings.add(model.z[exits[a][0][1]] - model.z[exits[b][0][1]] >= \
                                model.l_min - model.M*(1 - model.relations[a,b,4]))
            # down
            model.crossings.add(model.z[exits[a][0][0]] - model.z[exits[b][0][0]] <= \
                                model.M*(1 - model.relations[a,b,5]) - model.l_min)
            model.crossings.add(model.z[exits[a][0][0]] - model.z[exits[b][0][1]] <= \
                                model.M*(1 - model.relations[a,b,5]) - model.l_min)
            model.crossings.add(model.z[exits[a][0][1]] - model.z[exits[b][0][0]] <= \
                                model.M*(1 - model.relations[a,b,5]) - model.l_min)
            model.crossings.add(model.z[exits[a][0][1]] - model.z[exits[b][0][1]] <= \
                                model.M*(1 - model.relations[a,b,5]) - model.l_min)

    SolverFactory('cbc').solve(model, tee=True)
#    model.pprint()
    return model

if __name__=='__main__':
    main()
