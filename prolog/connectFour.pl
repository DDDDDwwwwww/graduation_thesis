role(red).
role(black).
base(cell(VarX, VarY, VarP)) :- x(VarX), y(VarY), role(VarP).
base(control(red)).
base(control(black)).
input(VarP, drop(VarX)) :- role(VarP), x(VarX).
input(VarP, noop) :- role(VarP).
init(control(red)).
legal(red, noop) :- true(control(black)).
legal(red, drop(VarX)) :- true(control(red)), columnopen(VarX).
legal(black, noop) :- true(control(red)).
legal(black, drop(VarX)) :- true(control(black)), columnopen(VarX).
next(cell(VarX, 1, VarPlayer)) :- does(VarPlayer, drop(VarX)), columnempty(VarX).
next(cell(VarX, VarY2, VarPlayer)) :- does(VarPlayer, drop(VarX)), cellopen(VarX, VarY2), succ(VarY1, VarY2), (\+ cellopen(VarX, VarY1)).
next(cell(VarX, VarY, VarPlayer)) :- true(cell(VarX, VarY, VarPlayer)).
next(control(red)) :- true(control(black)).
next(control(black)) :- true(control(red)).
terminal :- line(red).
terminal :- line(black).
terminal :- (\+ boardopen).
goal(red, 100) :- line(red).
goal(red, 50) :- (\+ line(red)), (\+ line(black)), (\+ boardopen).
goal(red, 0) :- line(black).
goal(red, 0) :- (\+ line(red)), (\+ line(black)), boardopen.
goal(black, 100) :- line(black).
goal(black, 50) :- (\+ line(red)), (\+ line(black)), (\+ boardopen).
goal(black, 0) :- line(red).
goal(black, 0) :- (\+ line(red)), (\+ line(black)), boardopen.
cellopen(VarX, VarY) :- x(VarX), y(VarY), (\+ true(cell(VarX, VarY, red))), (\+ true(cell(VarX, VarY, black))).
columnopen(VarX) :- cellopen(VarX, 6).
columnempty(VarX) :- cellopen(VarX, 1).
boardopen :- x(VarX), columnopen(VarX).
line(VarPlayer) :- true(cell(VarX1, VarY, VarPlayer)), succ(VarX1, VarX2), succ(VarX2, VarX3), succ(VarX3, VarX4), true(cell(VarX2, VarY, VarPlayer)), true(cell(VarX3, VarY, VarPlayer)), true(cell(VarX4, VarY, VarPlayer)).
line(VarPlayer) :- true(cell(VarX, VarY1, VarPlayer)), succ(VarY1, VarY2), succ(VarY2, VarY3), succ(VarY3, VarY4), true(cell(VarX, VarY2, VarPlayer)), true(cell(VarX, VarY3, VarPlayer)), true(cell(VarX, VarY4, VarPlayer)).
line(VarPlayer) :- true(cell(VarX1, VarY1, VarPlayer)), succ(VarX1, VarX2), succ(VarX2, VarX3), succ(VarX3, VarX4), succ(VarY1, VarY2), succ(VarY2, VarY3), succ(VarY3, VarY4), true(cell(VarX2, VarY2, VarPlayer)), true(cell(VarX3, VarY3, VarPlayer)), true(cell(VarX4, VarY4, VarPlayer)).
line(VarPlayer) :- true(cell(VarX1, VarY4, VarPlayer)), succ(VarX1, VarX2), succ(VarX2, VarX3), succ(VarX3, VarX4), succ(VarY3, VarY4), succ(VarY2, VarY3), succ(VarY1, VarY2), true(cell(VarX2, VarY3, VarPlayer)), true(cell(VarX3, VarY2, VarPlayer)), true(cell(VarX4, VarY1, VarPlayer)).
succ(1, 2).
succ(2, 3).
succ(3, 4).
succ(4, 5).
succ(5, 6).
succ(6, 7).
succ(7, 8).
x(1).
x(2).
x(3).
x(4).
x(5).
x(6).
x(7).
x(8).
y(1).
y(2).
y(3).
y(4).
y(5).
y(6).