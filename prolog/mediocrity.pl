role(a).
role(b).
role(c).
base(score(VarP, VarN)) :- role(VarP), num(VarN).
base(step(VarN)) :- lt(VarN, 6).
init(score(VarP, 0)) :- role(VarP).
init(step(0)).
input(VarP, VarN) :- role(VarP), lt(VarN, 9).
legal(VarP, VarN) :- role(VarP), lt(VarN, 9).
next(step(VarN)) :- true(step(VarM)), succ(VarM, VarN).
mediocre(VarP) :- does(VarP, VarL), does(VarQ, VarM), does(VarR, VarN), lt(VarM, VarL), lt(VarL, VarN).
next(score(VarP, VarN)) :- true(score(VarP, VarN)), (\+ mediocre(VarP)).
next(score(VarP, VarN)) :- true(score(VarP, VarM)), mediocre(VarP), does(VarP, VarL), add(VarN, VarM, VarL).
terminal :- true(step(5)).
winner(VarP) :- true(score(VarP, VarL)), true(score(VarQ, VarM)), true(score(VarR, VarN)), lt(VarM, VarL), lt(VarL, VarN).
goal(VarP, 100) :- winner(VarP).
goal(VarP, 0) :- role(VarP), (\+ winner(VarP)).
add(VarX, 0, VarX) :- num(VarX).
add(VarX, VarX, 0) :- num(VarX).
add(VarC, VarA, VarB) :- succ(VarA, VarAa), succ(VarBb, VarB), add(VarC, VarAa, VarBb), num(VarC).
lt(VarA, VarB) :- succ(VarA, VarB).
lt(VarA, VarB) :- succ(VarA, VarC), lt(VarC, VarB).
num(0).
num(1).
num(2).
num(3).
num(4).
num(5).
num(6).
num(7).
num(8).
num(9).
num(10).
num(11).
num(12).
num(13).
num(14).
num(15).
num(16).
num(17).
num(18).
num(19).
num(20).
num(21).
num(22).
num(23).
num(24).
num(25).
num(26).
num(27).
num(28).
num(29).
num(30).
num(31).
num(32).
num(33).
num(34).
num(35).
succ(0, 1).
succ(1, 2).
succ(2, 3).
succ(3, 4).
succ(4, 5).
succ(5, 6).
succ(6, 7).
succ(7, 8).
succ(8, 9).
succ(9, 10).
succ(10, 11).
succ(11, 12).
succ(12, 13).
succ(13, 14).
succ(14, 15).
succ(15, 16).
succ(16, 17).
succ(17, 18).
succ(18, 19).
succ(19, 20).
succ(20, 21).
succ(21, 22).
succ(22, 23).
succ(23, 24).
succ(24, 25).
succ(25, 26).
succ(26, 27).
succ(27, 28).
succ(28, 29).
succ(29, 30).
succ(30, 31).
succ(31, 32).
succ(32, 33).
succ(33, 34).
succ(34, 35).