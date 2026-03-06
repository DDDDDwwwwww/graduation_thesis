role(robot).
base(cell(a)).
base(cell(b)).
base(cell(c)).
base(cell(d)).
base(gold(a)).
base(gold(b)).
base(gold(c)).
base(gold(d)).
base(gold(i)).
base(step(1)).
base(step(VarX)) :- succ(VarY, VarX).
input(robot, move).
input(robot, grab).
input(robot, drop).
init(cell(a)).
init(gold(c)).
init(step(1)).
next(cell(VarY)) :- does(robot, move), true(cell(VarX)), adjacent(VarX, VarY).
next(cell(VarX)) :- does(robot, grab), true(cell(VarX)).
next(cell(VarX)) :- does(robot, drop), true(cell(VarX)).
next(gold(VarX)) :- does(robot, move), true(gold(VarX)).
next(gold(i)) :- does(robot, grab), true(cell(VarX)), true(gold(VarX)).
next(gold(i)) :- does(robot, grab), true(gold(i)).
next(gold(VarY)) :- does(robot, grab), true(cell(VarX)), true(gold(VarY)), VarX \= VarY.
next(gold(VarX)) :- does(robot, drop), true(cell(VarX)), true(gold(i)).
next(gold(VarX)) :- does(robot, drop), true(gold(VarX)), VarX \= i.
next(step(VarY)) :- true(step(VarX)), succ(VarX, VarY).
adjacent(a, b).
adjacent(b, c).
adjacent(c, d).
adjacent(d, a).
succ(1, 2).
succ(2, 3).
succ(3, 4).
succ(4, 5).
succ(5, 6).
succ(6, 7).
succ(7, 8).
succ(8, 9).
succ(9, 10).
legal(robot, move) :- succ(1, 2).
legal(robot, grab) :- true(cell(VarX)), true(gold(VarX)).
legal(robot, drop) :- true(gold(i)).
goal(robot, 100) :- true(gold(a)).
goal(robot, 0) :- true(gold(VarX)), VarX \= a.
terminal :- true(step(10)).
terminal :- true(gold(a)).