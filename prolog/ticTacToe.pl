role(xplayer).
role(oplayer).
index(1).
index(2).
index(3).
base(cell(VarX, VarY, b)) :- index(VarX), index(VarY).
base(cell(VarX, VarY, x)) :- index(VarX), index(VarY).
base(cell(VarX, VarY, o)) :- index(VarX), index(VarY).
base(control(VarP)) :- role(VarP).
input(VarP, mark(VarX, VarY)) :- index(VarX), index(VarY), role(VarP).
input(VarP, noop) :- role(VarP).
init(cell(1, 1, b)).
init(cell(1, 2, b)).
init(cell(1, 3, b)).
init(cell(2, 1, b)).
init(cell(2, 2, b)).
init(cell(2, 3, b)).
init(cell(3, 1, b)).
init(cell(3, 2, b)).
init(cell(3, 3, b)).
init(control(xplayer)).
next(cell(VarM, VarN, x)) :- does(xplayer, mark(VarM, VarN)), true(cell(VarM, VarN, b)).
next(cell(VarM, VarN, o)) :- does(oplayer, mark(VarM, VarN)), true(cell(VarM, VarN, b)).
next(cell(VarM, VarN, VarW)) :- true(cell(VarM, VarN, VarW)), VarW \= b.
next(cell(VarM, VarN, b)) :- does(VarW, mark(VarJ, VarK)), true(cell(VarM, VarN, b)), (VarM \= VarJ ; VarN \= VarK).
next(control(xplayer)) :- true(control(oplayer)).
next(control(oplayer)) :- true(control(xplayer)).
row(VarM, VarX) :- true(cell(VarM, 1, VarX)), true(cell(VarM, 2, VarX)), true(cell(VarM, 3, VarX)).
column(VarN, VarX) :- true(cell(1, VarN, VarX)), true(cell(2, VarN, VarX)), true(cell(3, VarN, VarX)).
diagonal(VarX) :- true(cell(1, 1, VarX)), true(cell(2, 2, VarX)), true(cell(3, 3, VarX)).
diagonal(VarX) :- true(cell(1, 3, VarX)), true(cell(2, 2, VarX)), true(cell(3, 1, VarX)).
line(VarX) :- row(VarM, VarX).
line(VarX) :- column(VarM, VarX).
line(VarX) :- diagonal(VarX).
open :- true(cell(VarM, VarN, b)).
legal(VarW, mark(VarX, VarY)) :- true(cell(VarX, VarY, b)), true(control(VarW)).
legal(xplayer, noop) :- true(control(oplayer)).
legal(oplayer, noop) :- true(control(xplayer)).
goal(xplayer, 100) :- line(x).
goal(xplayer, 50) :- (\+ line(x)), (\+ line(o)), (\+ open).
goal(xplayer, 0) :- line(o).
goal(oplayer, 100) :- line(o).
goal(oplayer, 50) :- (\+ line(x)), (\+ line(o)), (\+ open).
goal(oplayer, 0) :- line(x).
terminal :- line(x).
terminal :- line(o).
terminal :- (\+ open).