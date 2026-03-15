role(white).
role(black).
init(cell(1, 1, b)).
init(cell(1, 2, b)).
init(cell(1, 3, b)).
init(cell(1, 4, b)).
init(cell(1, 5, b)).
init(cell(1, 6, b)).
init(cell(1, 7, b)).
init(cell(1, 8, b)).
init(cell(2, 1, b)).
init(cell(2, 2, b)).
init(cell(2, 3, b)).
init(cell(2, 4, b)).
init(cell(2, 5, b)).
init(cell(2, 6, b)).
init(cell(2, 7, b)).
init(cell(2, 8, b)).
init(cell(3, 1, e)).
init(cell(3, 2, e)).
init(cell(3, 3, e)).
init(cell(3, 4, e)).
init(cell(3, 5, e)).
init(cell(3, 6, e)).
init(cell(3, 7, e)).
init(cell(3, 8, e)).
init(cell(4, 1, e)).
init(cell(4, 2, e)).
init(cell(4, 3, e)).
init(cell(4, 4, e)).
init(cell(4, 5, e)).
init(cell(4, 6, e)).
init(cell(4, 7, e)).
init(cell(4, 8, e)).
init(cell(5, 1, e)).
init(cell(5, 2, e)).
init(cell(5, 3, e)).
init(cell(5, 4, e)).
init(cell(5, 5, e)).
init(cell(5, 6, e)).
init(cell(5, 7, e)).
init(cell(5, 8, e)).
init(cell(6, 1, e)).
init(cell(6, 2, e)).
init(cell(6, 3, e)).
init(cell(6, 4, e)).
init(cell(6, 5, e)).
init(cell(6, 6, e)).
init(cell(6, 7, e)).
init(cell(6, 8, e)).
init(cell(7, 1, w)).
init(cell(7, 2, w)).
init(cell(7, 3, w)).
init(cell(7, 4, w)).
init(cell(7, 5, w)).
init(cell(7, 6, w)).
init(cell(7, 7, w)).
init(cell(7, 8, w)).
init(cell(8, 1, w)).
init(cell(8, 2, w)).
init(cell(8, 3, w)).
init(cell(8, 4, w)).
init(cell(8, 5, w)).
init(cell(8, 6, w)).
init(cell(8, 7, w)).
init(cell(8, 8, w)).
init(control(white)).
succ(1, 2).
succ(2, 3).
succ(3, 4).
succ(4, 5).
succ(5, 6).
succ(6, 7).
succ(7, 8).
adjacent_col(VarY, VarNy) :- succ(VarY, VarNy).
adjacent_col(VarY, VarNy) :- succ(VarNy, VarY).
legal(white, move(VarX, VarY, VarNx, VarY)) :- true(control(white)), true(cell(VarX, VarY, w)), succ(VarNx, VarX), true(cell(VarNx, VarY, e)).
legal(white, move(VarX, VarY, VarNx, VarNy)) :- true(control(white)), true(cell(VarX, VarY, w)), succ(VarNx, VarX), adjacent_col(VarY, VarNy), (\+ true(cell(VarNx, VarNy, w))).
legal(black, move(VarX, VarY, VarNx, VarY)) :- true(control(black)), true(cell(VarX, VarY, b)), succ(VarX, VarNx), true(cell(VarNx, VarY, e)).
legal(black, move(VarX, VarY, VarNx, VarNy)) :- true(control(black)), true(cell(VarX, VarY, b)), succ(VarX, VarNx), adjacent_col(VarY, VarNy), (\+ true(cell(VarNx, VarNy, b))).
next(control(white)) :- true(control(black)).
next(control(black)) :- true(control(white)).
next(cell(VarNx, VarNy, w)) :- does(white, move(VarX, VarY, VarNx, VarNy)).
next(cell(VarNx, VarNy, b)) :- does(black, move(VarX, VarY, VarNx, VarNy)).
next(cell(VarX, VarY, e)) :- does(VarPlayer, move(VarX, VarY, VarNx, VarNy)).
changed(VarX, VarY) :- does(VarPlayer, move(VarX, VarY, VarNx, VarNy)).
changed(VarNx, VarNy) :- does(VarPlayer, move(VarX, VarY, VarNx, VarNy)).
next(cell(VarX, VarY, VarState)) :- true(cell(VarX, VarY, VarState)), (\+ changed(VarX, VarY)).
white_wins :- true(cell(1, VarY, w)).
black_wins :- true(cell(8, VarY, b)).
has_legal_move :- role(VarPlayer), legal(VarPlayer, VarMove).
white_has_move :- legal(white, VarMove).
black_has_move :- legal(black, VarMove).
white_no_legal :- true(control(white)), (\+ white_has_move).
black_no_legal :- true(control(black)), (\+ black_has_move).
terminal :- white_wins.
terminal :- black_wins.
terminal :- (\+ has_legal_move).
goal(white, 100) :- white_wins.
goal(white, 100) :- black_no_legal.
goal(white, 0) :- black_wins.
goal(white, 0) :- white_no_legal.
goal(black, 100) :- black_wins.
goal(black, 100) :- white_no_legal.
goal(black, 0) :- white_wins.
goal(black, 0) :- black_no_legal.
goal(white, 50) :- (\+ white_wins), (\+ black_wins), (\+ white_no_legal), (\+ black_no_legal).
goal(black, 50) :- (\+ white_wins), (\+ black_wins), (\+ white_no_legal), (\+ black_no_legal).