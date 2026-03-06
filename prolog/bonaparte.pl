role(france).
role(germany).
role(russia).
base(spring).
base(autumn).
base(resource).
base(control(VarP, VarT)) :- role(VarP), supply(VarT).
base(army(VarP, VarT)) :- role(VarP), territory(VarT).
input(VarR, noop) :- role(VarR).
input(VarR, terminate) :- role(VarR).
input(VarR, disband(VarT)) :- role(VarR), territory(VarT).
input(VarR, disband2(VarT1, VarT2)) :- role(VarR), territory(VarT1), tlt(VarT1, VarT2).
input(VarR, disband3(VarT1, VarT2, VarT3)) :- role(VarR), territory(VarT1), tlt(VarT1, VarT2), tlt(VarT2, VarT3).
input(VarR, disband4(VarT1, VarT2, VarT3, VarT4)) :- role(VarR), territory(VarT1), tlt(VarT1, VarT2), tlt(VarT2, VarT3), tlt(VarT3, VarT4).
input(VarR, build1(VarS)) :- home(VarR, VarS).
input(VarR, build2(VarS, VarT)) :- home(VarR, VarS), home(VarR, VarT), slt(VarS, VarT).
inp(VarFrom, move(VarFrom, VarTo)) :- adjacent(VarFrom, VarTo).
inp(VarFrom, support_hold(VarFrom, VarTo)) :- adjacent(VarFrom, VarTo).
inp(VarFrom, support_move(VarFrom, VarAttack_from, VarTo)) :- adjacent(VarAttack_from, VarTo), adjacent(VarFrom, VarTo).
input(VarP, VarM) :- role(VarP), inp(VarT, VarM).
input(VarP, combine2(VarM1, VarM2)) :- role(VarP), inp(VarT1, VarM1), inp(VarT2, VarM2), tlt(VarT1, VarT2).
input(VarP, combine3(VarM1, VarM2, VarM3)) :- role(VarP), inp(VarT1, VarM1), inp(VarT2, VarM2), inp(VarT3, VarM3), tlt(VarT1, VarT2), tlt(VarT2, VarT3).
init(spring).
init(year(1)).
init(control(VarP, VarS)) :- home(VarP, VarS), role(VarP), supply(VarS).
init(army(VarP, VarS)) :- home(VarP, VarS), role(VarP), supply(VarS).
next(autumn) :- true(spring).
next(resource) :- true(autumn).
next(spring) :- true(resource).
moving :- true(spring).
moving :- true(autumn).
next(year(VarN)) :- true(year(VarM)), succ(VarM, VarN), true(resource).
next(year(VarN)) :- true(year(VarN)), moving.
occupied(VarS) :- true(army(VarP1, VarS)), true(control(VarP2, VarS)), VarP1 \= VarP2.
in_control(VarP, VarS) :- true(control(VarP, VarS)), (\+ occupied(VarS)).
in_control(VarP, VarS) :- role(VarP), supply(VarS), true(army(VarP, VarS)).
next(control(VarP, VarS)) :- true(control(VarP, VarS)), moving.
next(control(VarP, VarS)) :- in_control(VarP, VarS), true(resource).
terminal :- true(year(30)), true(resource).
terminal :- lt(5, VarN), controls(VarPlayer, VarN), true(resource).
count_supply(VarP, bel, 0) :- role(VarP), (\+ in_control(VarP, bel)).
count_supply(VarP, bel, 1) :- in_control(VarP, bel).
count_supply(VarP, VarT, VarM) :- (\+ in_control(VarP, VarT)), senum(VarS, VarT), count_supply(VarP, VarS, VarM).
count_supply(VarP, VarT, VarN) :- in_control(VarP, VarT), senum(VarS, VarT), succ(VarM, VarN), count_supply(VarP, VarS, VarM).
controls(VarP, VarN) :- count_supply(VarP, war, VarN).
count_armies(VarP, bel, 0) :- role(VarP), (\+ true(army(VarP, bel))).
count_armies(VarP, bel, 1) :- true(army(VarP, bel)).
count_armies(VarP, VarT, VarN) :- (\+ true(army(VarP, VarT))), tenum(VarS, VarT), count_armies(VarP, VarS, VarN).
count_armies(VarP, VarT, VarN) :- true(army(VarP, VarT)), tenum(VarS, VarT), succ(VarM, VarN), count_armies(VarP, VarS, VarM).
armies(VarP, VarN) :- count_armies(VarP, war, VarN).
goal(VarP, 100) :- controls(VarP, VarN), lt(5, VarN).
goal(VarP, 50) :- controls(VarP, VarN), controls(VarP2, VarM), controls(VarP3, VarL), VarP \= VarP2, VarP \= VarP3, VarP2 \= VarP3, lt(VarN, 6), lt(VarM, VarN), lt(VarL, VarN).
goal(VarP, 20) :- controls(VarP, VarN), controls(VarP2, VarN), controls(VarP3, VarM), VarP \= VarP2, lt(VarM, VarN).
goal(VarP, 10) :- controls(VarP, VarN), controls(VarP2, VarN), controls(VarP3, VarN), VarP \= VarP2, VarP \= VarP3, VarP2 \= VarP3.
goal(VarP, 0) :- controls(VarP, VarN), controls(VarP2, VarM), VarP \= VarP2, lt(VarN, VarM).
legal(VarP, noop) :- armies(VarP, VarM), controls(VarP, VarN), le(VarM, VarN), true(resource).
legal(VarP, build1(VarS)) :- in_control(VarP, VarS), home(VarP, VarS), controls(VarP, VarN), armies(VarP, VarM), lt(VarM, VarN), (\+ true(army(VarP, VarS))), true(resource).
legal(VarP, build2(VarS, VarT)) :- in_control(VarP, VarS), in_control(VarP, VarT), home(VarP, VarS), home(VarP, VarT), slt(VarS, VarT), controls(VarP, VarN), armies(VarP, VarM), lt(VarM, VarL), succ(VarL, VarN), (\+ true(army(VarP, VarS))), (\+ true(army(VarP, VarT))), true(resource).
legal(VarP, disband(VarT)) :- true(army(VarP, VarT)), controls(VarP, VarN), lt(0, VarN), armies(VarP, VarM), succ(VarN, VarM), true(resource).
legal(VarP, disband2(VarS, VarT)) :- true(army(VarP, VarS)), true(army(VarP, VarT)), controls(VarP, VarN), lt(0, VarN), armies(VarP, VarM), succ(VarN, VarX), succ(VarX, VarM), tlt(VarS, VarT), true(resource).
legal(VarP, disband3(VarS, VarT, VarU)) :- true(army(VarP, VarS)), true(army(VarP, VarT)), true(army(VarP, VarU)), controls(VarP, VarN), lt(0, VarN), armies(VarP, VarM), tlt(VarS, VarT), tlt(VarT, VarU), succ(VarN, VarX), succ(VarX, VarY), succ(VarY, VarM), true(resource).
legal(VarP, disband4(VarS, VarT, VarU, VarV)) :- true(army(VarP, VarS)), true(army(VarP, VarT)), true(army(VarP, VarU)), true(army(VarP, VarV)), controls(VarP, VarN), lt(0, VarN), armies(VarP, VarM), tlt(VarS, VarT), tlt(VarT, VarU), tlt(VarU, VarV), succ(VarN, VarX), succ(VarX, VarY), succ(VarY, VarZ), succ(VarZ, VarM), true(resource).
legal(VarP, terminate) :- controls(VarP, 0), armies(VarP, VarM), lt(0, VarM), true(resource).
disbanded(VarT) :- does(VarP, disband(VarT)).
disbanded(VarT) :- does(VarP, disband2(VarT, VarU)).
disbanded(VarT) :- does(VarP, disband2(VarU, VarT)).
disbanded(VarT) :- does(VarP, disband3(VarT, VarU, VarV)).
disbanded(VarT) :- does(VarP, disband3(VarU, VarT, VarV)).
disbanded(VarT) :- does(VarP, disband3(VarU, VarV, VarT)).
disbanded(VarT) :- does(VarP, disband4(VarT, VarU, VarV, VarW)).
disbanded(VarT) :- does(VarP, disband4(VarU, VarT, VarV, VarW)).
disbanded(VarT) :- does(VarP, disband4(VarU, VarV, VarT, VarW)).
disbanded(VarT) :- does(VarP, disband4(VarU, VarV, VarW, VarT)).
disbanded(VarT) :- true(army(VarP, VarT)), does(VarP, terminate).
next(army(VarP, VarT)) :- true(army(VarP, VarT)), true(resource), (\+ disbanded(VarT)).
next(army(VarP, VarT)) :- does(VarP, build1(VarT)), true(resource).
next(army(VarP, VarT)) :- does(VarP, build2(VarT, VarS)), true(resource).
next(army(VarP, VarT)) :- does(VarP, build2(VarS, VarT)), true(resource).
legal(VarP, noop) :- role(VarP), moving.
lgl(VarP, VarFrom, move(VarFrom, VarTo)) :- true(army(VarP, VarFrom)), adjacent(VarFrom, VarTo), moving.
lgl(VarP, VarFrom, support_hold(VarFrom, VarTo)) :- true(army(VarP, VarFrom)), true(army(VarAnyone, VarTo)), role(VarAnyone), adjacent(VarFrom, VarTo), moving.
lgl(VarP, VarFrom, support_move(VarFrom, VarAttack_from, VarTo)) :- true(army(VarP, VarFrom)), true(army(VarAny, VarAttack_from)), adjacent(VarFrom, VarTo), adjacent(VarAttack_from, VarTo), VarFrom \= VarAttack_from, moving.
legal(VarP, VarMove) :- lgl(VarP, VarI, VarMove).
ordered_move(VarP, VarFrom, VarTo) :- does(VarP, move(VarFrom, VarTo)).
ordered_support(VarP, VarFrom, VarTo) :- does(VarP, support_hold(VarFrom, VarTo)).
ordered_attack_support(VarP, VarFrom, VarAf, VarTo) :- does(VarP, support_move(VarFrom, VarAf, VarTo)).
legal(VarP, combine2(VarMove1, VarMove2)) :- lgl(VarP, VarI, VarMove1), lgl(VarP, VarJ, VarMove2), tlt(VarI, VarJ).
ordered_move(VarP, VarFrom, VarTo) :- does(VarP, combine2(move(VarFrom, VarTo), VarOther)).
ordered_move(VarP, VarFrom, VarTo) :- does(VarP, combine2(VarOther, move(VarFrom, VarTo))).
ordered_support(VarP, VarFrom, VarTo) :- does(VarP, combine2(support_hold(VarFrom, VarTo), VarOther)).
ordered_support(VarP, VarFrom, VarTo) :- does(VarP, combine2(VarOther, support_hold(VarFrom, VarTo))).
ordered_attack_support(VarP, VarFrom, VarAf, VarTo) :- does(VarP, combine2(support_move(VarFrom, VarAf, VarTo), VarOther)).
ordered_attack_support(VarP, VarFrom, VarAf, VarTo) :- does(VarP, combine2(VarOther, support_move(VarFrom, VarAf, VarTo))).
legal(VarP, combine3(VarMove1, VarMove2, VarMove3)) :- lgl(VarP, VarI, VarMove1), lgl(VarP, VarJ, VarMove2), lgl(VarP, VarK, VarMove3), tlt(VarI, VarJ), tlt(VarJ, VarK).
ordered_move(VarP, VarFrom, VarTo) :- does(VarP, combine3(move(VarFrom, VarTo), VarOther, VarOther2)).
ordered_move(VarP, VarFrom, VarTo) :- does(VarP, combine3(VarOther, move(VarFrom, VarTo), VarOther2)).
ordered_move(VarP, VarFrom, VarTo) :- does(VarP, combine3(VarOther, VarOther2, move(VarFrom, VarTo))).
ordered_support(VarP, VarFrom, VarTo) :- does(VarP, combine3(support_hold(VarFrom, VarTo), VarOther, VarOther2)).
ordered_support(VarP, VarFrom, VarTo) :- does(VarP, combine3(VarOther, support_hold(VarFrom, VarTo), VarOther2)).
ordered_support(VarP, VarFrom, VarTo) :- does(VarP, combine3(VarOther, VarOther2, support_hold(VarFrom, VarTo))).
ordered_attack_support(VarP, VarFrom, VarAf, VarTo) :- does(VarP, combine3(support_move(VarFrom, VarAf, VarTo), VarOther, VarOther2)).
ordered_attack_support(VarP, VarFrom, VarAf, VarTo) :- does(VarP, combine3(VarOther, support_move(VarFrom, VarAf, VarTo), VarOther2)).
ordered_attack_support(VarP, VarFrom, VarAf, VarTo) :- does(VarP, combine3(VarOther, VarOther2, support_move(VarFrom, VarAf, VarTo))).
support_void(VarT) :- true(army(VarP, VarT)), ordered_move(VarP2, VarFrom, VarT), VarP \= VarP2, territory(VarAf), (\+ ordered_attack_support(VarP, VarT, VarAf, VarFrom)).
supported_def(VarP, VarFrom, VarT) :- true(army(VarP, VarT)), ordered_support(VarP2, VarFrom, VarT).
defended(VarT, bel, 2) :- true(army(VarP, VarT)), (\+ support_void(bel)), supported_def(VarP, bel, VarT).
defended(VarT, bel, 1) :- true(army(VarP, VarT)), (\+ supported_def(VarP, bel, VarT)).
defended(VarT, bel, 1) :- true(army(VarP, VarT)), support_void(bel).
defended(VarT, VarFromt, VarN) :- true(army(VarP, VarT)), (\+ support_void(VarFromt)), supported_def(VarP, VarFromt, VarT), tenum(VarPrev, VarFromt), succ(VarM, VarN), defended(VarT, VarPrev, VarM).
defended(VarT, VarFromt, VarM) :- true(army(VarP, VarT)), (\+ supported_def(VarP, VarFromt, VarT)), tenum(VarPrev, VarFromt), defended(VarT, VarPrev, VarM).
defended(VarT, VarFromt, VarM) :- true(army(VarP, VarT)), support_void(VarFromt), tenum(VarPrev, VarFromt), defended(VarT, VarPrev, VarM).
defense_strength(VarT, VarN) :- true(army(VarP, VarT)), ordered_move(VarP2, VarAny, VarT), VarP \= VarP2, defended(VarT, war, VarN).
supported_att(VarP, VarFrom, VarAf, VarTo) :- ordered_move(VarP, VarAny, VarT0), ordered_attack_support(VarP2, VarFrom, VarAf, VarTo).
attacked(VarP, bel, bel, VarTo, 1) :- ordered_move(VarP, bel, VarTo).
attacked(VarP, bel, VarAf, VarTo, 1) :- territory(VarAf), (\+ support_void(bel)), supported_att(VarP, bel, VarAf, VarTo).
attacked(VarP, bel, VarAf, VarTo, 0) :- territory(VarAf), role(VarP), territory(VarTo), (\+ true(army(VarP, bel))), (\+ supported_att(VarP, bel, VarAf, VarTo)).
attacked(VarP, bel, VarAf, VarTo, 0) :- territory(VarAf), support_void(bel), supported_att(VarP, bel, VarAf, VarTo).
attacked(VarP, VarFrom, VarAf, VarTo, VarN) :- territory(VarAf), ordered_move(VarP, VarFrom, VarTo), tenum(VarPrev, VarFrom), succ(VarM, VarN), attacked(VarP, VarPrev, VarAf, VarTo, VarM).
attacked(VarP, VarFrom, VarAf, VarTo, VarN) :- (\+ support_void(VarFrom)), supported_att(VarP, VarFrom, VarAf, VarTo), tenum(VarPrev, VarFrom), succ(VarM, VarN), attacked(VarP, VarPrev, VarAf, VarTo, VarM).
attacked(VarP, VarFrom, VarAf, VarTo, VarM) :- territory(VarAf), (\+ ordered_move(VarP, VarFrom, VarTo)), (\+ supported_att(VarP, VarFrom, VarAf, VarTo)), tenum(VarPrev, VarFrom), attacked(VarP, VarPrev, VarAf, VarTo, VarM).
attacked(VarP, VarFrom, VarAf, VarTo, VarM) :- support_void(VarFrom), supported_att(VarP, VarFrom, VarAf, VarTo), tenum(VarPrev, VarFrom), attacked(VarP, VarPrev, VarAf, VarTo, VarM).
attack_strength(VarP, VarAf, VarTo, VarN) :- ordered_move(VarP, VarAf, VarTo), attacked(VarP, war, VarAf, VarTo, VarN).
destroyed(VarT) :- true(army(VarP, VarT)), attack_strength(VarP2, VarAf, VarT, VarAtt), defense_strength(VarT, VarDef), lt(VarDef, VarAtt).
strength_matched(VarP, VarTo) :- attack_strength(VarP, VarAf, VarTo, VarN), attack_strength(VarP2, VarAf2, VarTo, VarN2), le(VarN, VarN2), VarP \= VarP2.
strength_matched(VarP, VarTo) :- ordered_move(VarP, VarFrom, VarTo), ordered_move(VarP, VarFrom2, VarTo), VarFrom \= VarFrom2.
victorious(VarP, VarFrom, VarTo) :- ordered_move(VarP, VarFrom, VarTo), (\+ strength_matched(VarP, VarTo)).
contested(VarTo) :- ordered_move(VarP, VarFrom, VarTo), ordered_move(VarP2, VarFrom2, VarTo), VarFrom \= VarFrom2.
move_possible(VarP, VarFrom, VarTo) :- ordered_move(VarP, VarFrom, VarTo), (\+ contested(VarTo)), now_empty(VarTo), role(VarP2).
move_possible(VarP, VarFrom, VarTo) :- ordered_move(VarP, VarFrom, VarTo), victorious(VarP, VarFrom, VarTo), now_empty(VarTo), role(VarP2).
occupied(VarT) :- role(VarP), true(army(VarP, VarT)).
now_empty(VarT) :- territory(VarT), destroyed(VarT).
now_empty(VarT) :- role(VarP), territory(VarT), territory(VarAnywhere), move_possible(VarP, VarT, VarAnywhere).
now_empty(VarT) :- territory(VarT), (\+ occupied(VarT)).
next(army(VarP, VarTo)) :- move_possible(VarP, VarFrom, VarTo), moving.
next(army(VarP, VarT)) :- true(army(VarP, VarT)), (\+ now_empty(VarT)), moving.
adjacent(bre, gas).
adjacent(bre, par).
adjacent(bre, pic).
adjacent(gas, par).
adjacent(gas, mar).
adjacent(par, pic).
adjacent(bur, gas).
adjacent(bur, mar).
adjacent(bur, par).
adjacent(bur, pic).
adjacent(bur, ruh).
adjacent(bur, mun).
adjacent(bel, bur).
adjacent(bel, pic).
adjacent(bel, hol).
adjacent(bel, ruh).
adjacent(hol, kie).
adjacent(hol, ruh).
adjacent(mar, pie).
adjacent(kie, ruh).
adjacent(kie, mun).
adjacent(ber, kie).
adjacent(ber, mun).
adjacent(ber, pru).
adjacent(ber, sil).
adjacent(mun, ruh).
adjacent(mun, tyr).
adjacent(mun, sil).
adjacent(pru, sil).
adjacent(pru, war).
adjacent(sil, war).
adjacent(pie, tyr).
adjacent(pie, tus).
adjacent(pie, ven).
adjacent(tyr, vie).
adjacent(boh, sil).
adjacent(boh, gal).
adjacent(boh, vie).
adjacent(boh, tyr).
adjacent(boh, mun).
adjacent(bud, gal).
adjacent(bud, tri).
adjacent(bud, vie).
adjacent(tri, tyr).
adjacent(tri, vie).
adjacent(gal, war).
adjacent(gal, ukr).
adjacent(gal, vie).
adjacent(gal, sil).
adjacent(lvn, stp).
adjacent(lvn, mos).
adjacent(lvn, war).
adjacent(lvn, pru).
adjacent(mos, stp).
adjacent(mos, sev).
adjacent(mos, ukr).
adjacent(mos, war).
adjacent(sev, ukr).
adjacent(ukr, war).
adjacent(gas, bre).
adjacent(par, bre).
adjacent(pic, bre).
adjacent(par, gas).
adjacent(mar, gas).
adjacent(pic, par).
adjacent(gas, bur).
adjacent(mar, bur).
adjacent(par, bur).
adjacent(pic, bur).
adjacent(ruh, bur).
adjacent(mun, bur).
adjacent(bur, bel).
adjacent(pic, bel).
adjacent(hol, bel).
adjacent(ruh, bel).
adjacent(kie, hol).
adjacent(ruh, hol).
adjacent(pie, mar).
adjacent(ruh, kie).
adjacent(mun, kie).
adjacent(kie, ber).
adjacent(mun, ber).
adjacent(pru, ber).
adjacent(sil, ber).
adjacent(ruh, mun).
adjacent(tyr, mun).
adjacent(sil, mun).
adjacent(sil, pru).
adjacent(war, pru).
adjacent(war, sil).
adjacent(tyr, pie).
adjacent(tus, pie).
adjacent(ven, pie).
adjacent(vie, tyr).
adjacent(sil, boh).
adjacent(gal, boh).
adjacent(vie, boh).
adjacent(tyr, boh).
adjacent(mun, boh).
adjacent(gal, bud).
adjacent(tri, bud).
adjacent(vie, bud).
adjacent(tyr, tri).
adjacent(vie, tri).
adjacent(war, gal).
adjacent(ukr, gal).
adjacent(vie, gal).
adjacent(sil, gal).
adjacent(stp, lvn).
adjacent(mos, lvn).
adjacent(war, lvn).
adjacent(pru, lvn).
adjacent(stp, mos).
adjacent(sev, mos).
adjacent(ukr, mos).
adjacent(war, mos).
adjacent(ukr, sev).
adjacent(war, ukr).
territory(bel).
territory(ber).
territory(boh).
territory(bre).
territory(bud).
territory(bur).
territory(gal).
territory(gas).
territory(hol).
territory(kie).
territory(lvn).
territory(mar).
territory(mos).
territory(mun).
territory(par).
territory(pic).
territory(pie).
territory(pru).
territory(ruh).
territory(sev).
territory(sil).
territory(stp).
territory(swe).
territory(tri).
territory(tus).
territory(tyr).
territory(ukr).
territory(ven).
territory(vie).
territory(war).
tenum(bel, ber).
tenum(ber, boh).
tenum(boh, bre).
tenum(bre, bud).
tenum(bud, bur).
tenum(bur, gal).
tenum(gal, gas).
tenum(gas, hol).
tenum(hol, kie).
tenum(kie, lvn).
tenum(lvn, mar).
tenum(mar, mos).
tenum(mos, mun).
tenum(mun, par).
tenum(par, pic).
tenum(pic, pie).
tenum(pie, pru).
tenum(pru, ruh).
tenum(ruh, sev).
tenum(sev, sil).
tenum(sil, stp).
tenum(stp, swe).
tenum(swe, tri).
tenum(tri, tus).
tenum(tus, tyr).
tenum(tyr, ukr).
tenum(ukr, ven).
tenum(ven, vie).
tenum(vie, war).
tlt(VarA, VarB) :- tenum(VarA, VarB).
tlt(VarA, VarB) :- tenum(VarA, VarX), tlt(VarX, VarB).
supply(bel).
supply(ber).
supply(bud).
supply(hol).
supply(mar).
supply(mos).
supply(mun).
supply(par).
supply(ven).
supply(vie).
supply(war).
senum(bel, ber).
senum(ber, bud).
senum(bud, hol).
senum(hol, mar).
senum(mar, mos).
senum(mos, mun).
senum(mun, par).
senum(par, ven).
senum(ven, vie).
senum(vie, war).
slt(VarA, VarB) :- senum(VarA, VarB).
slt(VarA, VarB) :- senum(VarA, VarX), slt(VarX, VarB).
home(france, par).
home(france, mar).
home(germany, ber).
home(germany, mun).
home(russia, mos).
home(russia, war).
lt(VarA, VarB) :- succ(VarA, VarB).
lt(VarA, VarB) :- succ(VarA, VarC), lt(VarC, VarB).
le(VarA, VarA) :- num(VarA).
le(VarA, VarB) :- lt(VarA, VarB).
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