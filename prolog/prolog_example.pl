/* =========================================================
   Time Orb Adventure
   Implemented with SWI-Prolog + Planning.Domains online planner

   Main Features:
   - 4x4 (16) grid map, positions 1..16
   - Time Orb and Exit are initialized randomly and never in the same position
   - Player / Enemy only know if a cell contains the Time Orb / Exit after stepping into it
   - Player / Enemy cannot see each other's positions; they only "encounter" when on the same cell
   - When encountering:
       * If someone has the Orb -> robbery (with cooldown)
       * If no one has the Orb -> non-lethal scuffle
   - Scuffle uses history_state/2 to store previous round's positions:
       * 50% chance the enemy pushes the player back to the player's previous room
       * 50% chance the player pushes the enemy back to the enemy's previous room
       * If no history exists, both are randomly scattered
   - After robbery, both sides are randomly scattered (not onto the same cell or the exit),
     and enter a 3-round cooldown
   - Player starts with two items: time_shard, map_fragment
       * time_shard: return both player and enemy to their previous turn's positions (single-use)
       * map_fragment: predicts the enemy’s next step (based on the planner’s first action, reusable)
   - Player / Enemy goals:
       At start: pick up the Orb → reach exit to escape;
       When one holds the Orb → the other becomes “chaser mode”
       When someone escapes with the Orb → game ends.

   Player input examples (Prolog command line):
     ?- play.
     > help.
     > map.
     > n.
     > use(time_shard).
     > use(map_fragment).
     > inventory.
     > quit.
========================================================= */
:- encoding(utf8).

:- use_module(library(lists)).
:- use_module(library(random)).
:- use_module(library(readutil)).
:- use_module(library(http/http_open)).
:- use_module(library(http/json)).
:- use_module(library(http/http_client)).

:- dynamic here/1.
:- dynamic enemy_here/1.
:- dynamic orb_at/1.
:- dynamic exit_at/1.
:- dynamic has_orb/1.           % has_orb(player) / has_orb(enemy)
:- dynamic cooldown/1.          % cooldown rounds for robbery
:- dynamic turn/1.              % number of rounds passed
:- dynamic history_state/2.     % history_state(PlayerPrev,EnemyPrev)
:- dynamic owns/2.              % owns(player, Item)
:- dynamic current_plan/1.      % current_plan([ActionAtom,...])
:- dynamic adversary_mode/1.    % thief | chaser
:- dynamic forced_enemy_move/1.   % Cache grid for the enemy's movement when fall back to random movement.
:- dynamic history_orb_holder/1.   % previous turn orb holder: player / enemy / none
:- dynamic history_orb_at/1.      % The orb's position on the map during the previous round; none indicates it is not on the ground.


/* ===================== Entry ===================== */

play :-
    init_game,
    instructions,
    game_loop.

instructions :-
    nl,
    % write('=== Time Orb Escape ==='), nl,
    write('Command Examples:'), nl,
    write('  n.  s.  e.  w.                % Move north/south/east/west'), nl,
    write('  map.                          % View Map (Shows only your location)'), nl,
    write('  look.                         % View Current Location ID'), nl,
    write('  inventory.                    % View inventory'), nl,
    write('  use(time_shard).              % Use Time Shard to return to the previous turn''s position.'), nl,
    write('  use(map_fragment).            % Use map fragments to predict the enemy''s next move.'), nl,
    write('  help.                         % Display Command example again'), nl,
    write('  quit.                         % Exit the game'), nl, nl,
    describe_location,
    draw_map.

/* ===================== Initialization ===================== */

init_game :-
    % Clear previous game state
    retractall(here(_)),
    retractall(enemy_here(_)),
    retractall(orb_at(_)),
    retractall(exit_at(_)),
    retractall(has_orb(_)),
    retractall(cooldown(_)),
    retractall(turn(_)),
    retractall(history_state(_,_)),
    retractall(owns(_,_)),
    retractall(current_plan(_)),
    retractall(adversary_mode(_)),
    retractall(forced_enemy_move(_)),
    retractall(history_orb_holder(_)),
    retractall(history_orb_at(_)),

    % Randomly initialize 4 distinct positions: orb, exit, player, enemy
    findall(Pos, between(1,16,Pos), All),
    random_permutation(All, [Orb,Exit,Player,Enemy|_]),
    Orb \= Exit,

    asserta(orb_at(Orb)),
    asserta(exit_at(Exit)),
    asserta(here(Player)),
    asserta(enemy_here(Enemy)),

    % Initial items
    asserta(owns(player, time_shard)),
    asserta(owns(player, map_fragment)),

    % Initial state
    asserta(has_orb(none)),
    asserta(cooldown(0)),
    asserta(turn(0)),
    asserta(history_state(Player,Enemy)),
    asserta(adversary_mode(thief)),

    % Initial game display
    nl,
    write('=== Time Orb Adventure ==='), nl.
    % write('=== Time Orb Adventure ==='), nl,
    % debug_show_enemy_pos.   % debug part
    
/* ===================== Main Game Loop ===================== */

game_loop :-
    (   check_end
    ->  true
    ;   write('> '),
        read(UserCmd),
        here(OldPos),

        (   call_command(UserCmd)
        ->  true
        ;   write('I don''t understand this command. Try help. to find available commands.'), nl
        ),

        (   is_move_command(UserCmd),
            here(NewPos),
            NewPos =\= OldPos
        ->  enemy_turn,
            % debug_show_enemy_pos,
            advance_turn
        ;   true
        ),

        game_loop
    ).

/* ===================== Command Dispatcher ===================== */

call_command(help) :- !, instructions.
call_command(map)  :- !, draw_map.
call_command(look) :- !, describe_location.
call_command(inventory) :- !, show_inventory.
call_command(quit) :-
    !, write('You have chosen to exit the game.'), nl, halt.

% Movement shorthand
call_command(n) :- !, move_player(north).
call_command(s) :- !, move_player(south).
call_command(e) :- !, move_player(east).
call_command(w) :- !, move_player(west).

call_command(go(Dir)) :- !, move_player(Dir).
call_command(use(Item)) :- !, use_item(Item).

call_command(Other) :-
    format('Unknown command: ~w~n', [Other]),
    fail.

% Check whether this is a movement command
is_move_command(n).
is_move_command(s).
is_move_command(e).
is_move_command(w).
is_move_command(go(_)).

/* ===================== Map & Location ===================== */

describe_location :-
    here(Pos),
    format('Your current location is ~w (1-16 Sixteen-Grid). ~n', [Pos]),
    ( orb_at(Pos) ->
        write('You''ve discovered a mysterious glowing orb that looks like a Time Orb! The Time Orb will be automatically picked up.'), nl
    ; true),

    ( exit_at(Pos) ->
        write('This appears to be the exit. Bring the Time Orb here to escape.'), nl
    ; true).

draw_map :-
    here(P),
    nl, write('Map (4x4): Player position marked as [P], other positions are [..], grid numbers are for your reference only:'), nl,
    draw_row(1, P),
    draw_row(5, P),
    draw_row(9, P),
    draw_row(13, P),
    nl.

draw_row(Start, P) :-
    End is Start + 3,
    forall(between(Start, End, Pos),
           ( (Pos =:= P ->
                 format('[P ] ')
             ;
                 format('[~|~`0t~d~2+] ', [Pos])
             ))),
    nl.

/* Valid movement check */

move_player(Direction) :-
    here(OldPos),
    compute_new_pos(OldPos, Direction, NewPos),
    ( valid_step(OldPos, NewPos) ->
        update_history,
        retract(here(OldPos)),
        asserta(here(NewPos)),
        write('You have moved to the position'), write(NewPos), nl,
        after_player_move
    ; write('That way is blocked.'), nl
    ).

compute_new_pos(Pos, north, New) :- !, New is Pos - 4.
compute_new_pos(Pos, south, New) :- !, New is Pos + 4.
compute_new_pos(Pos, east,  New) :- !, New is Pos + 1.
compute_new_pos(Pos, west,  New) :- !, New is Pos - 1.
compute_new_pos(Pos, Dir, Pos) :-
    % Fallback: unknown direction
    format('Unknown Direction: ~w, keep still. ~n', [Dir]).

valid_step(Old, New) :-
    New >= 1, New =< 16,
    same_row_or_column(Old, New).

same_row_or_column(Old, New) :-
    % North/South: same column; East/West: same row
    RowOld is ((Old - 1) // 4),
    ColOld is ((Old - 1) mod 4),
    RowNew is ((New - 1) // 4),
    ColNew is ((New - 1) mod 4),
    ( RowOld =:= RowNew ; ColOld =:= ColNew ).

/* ===================== History & Time Flow ===================== */

update_history :-
    here(P), enemy_here(E),
    % save position
    retractall(history_state(_,_)),
    asserta(history_state(P,E)),
    % save orb ownership from the previous round
    has_orb(Holder),                 % player / enemy / none
    retractall(history_orb_holder(_)),
    asserta(history_orb_holder(Holder)),
    % save whether the orb from the previous round is on the ground & its position
    (   orb_at(Pos)
    ->  OrbLoc = Pos
    ;   OrbLoc = none
    ),
    retractall(history_orb_at(_)),
    asserta(history_orb_at(OrbLoc)).


advance_turn :-
    retract(turn(T)),
    T1 is T + 1,
    asserta(turn(T1)),
    decrement_cooldown.

decrement_cooldown :-
    cooldown(C),
    ( C > 0 ->
        C1 is C - 1,
        retract(cooldown(C)),
        asserta(cooldown(C1))
    ; true).

/* ===================== Player Item System ===================== */

show_inventory :-
    findall(Item, owns(player, Item), Items),
    write('Your inventory:'), nl,
    ( Items = [] ->
        write(' nothing in it.'), nl
    ; forall(member(I, Items),
             ( format('  - ~w~n', [I]) ))
    ).

% use_item(time_shard) :-
%     owns(player, time_shard),
%     !,
%     ( history_state(PPrev,EPrev) ->
%         retract(here(_)), asserta(here(PPrev)),
%         retract(enemy_here(_)), asserta(enemy_here(EPrev)),
%         retract(owns(player, time_shard)),
%         write('Time shard shatters into fragments, pulling you and your foe back to your positions from the previous round.'), nl,
%         draw_map
%         % debug_show_enemy_pos
%     ; write('There is no historical position to trace back to; the time fragments have not taken effect.'), nl
%     ).
use_item(time_shard) :-
    owns(player, time_shard),
    !,
    (   history_state(PPrev, EPrev),
        history_orb_holder(PrevHolder),
        history_orb_at(PrevOrbLoc)
    ->  % restore orb
        retract(here(_)),        asserta(here(PPrev)),
        retract(enemy_here(_)),  asserta(enemy_here(EPrev)),

        retractall(has_orb(_)),
        asserta(has_orb(PrevHolder)),

        retractall(orb_at(_)),
        (   PrevOrbLoc = none
        ->  true
        ;   asserta(orb_at(PrevOrbLoc))
        ),

        (   PrevHolder = player
        ->  (   owns(player, time_orb)
            ->  true
            ;   asserta(owns(player, time_orb))
            )
        ;   
            retractall(owns(player, time_orb))
        ),

        retract(owns(player, time_shard)),
        write('Time shard shatters into fragments, pulling you and your foe back to your positions from the previous round.'), nl,
        write('The orb will also return to its previous position.'), nl,
        draw_map
    ;   write('There is no historical state to rewind to; the time fragments have no effect.'), nl
    ).

use_item(time_shard) :-
    write('You have no spare time shard left.'), nl.

use_item(map_fragment) :-
    owns(player, map_fragment),
    !,
    retract(owns(player, map_fragment)), 
    predict_enemy_next(Info),
    format('The map fragment glowed faintly, revealing: ~w~n', [Info]).
use_item(map_fragment) :-
    write('You have no map fragments.'), nl.

use_item(Other) :-
    format('You don''t know how to use ~w. ~n', [Other]).

/* ===================== Orb & Exit Logic ===================== */

% after_player_move :-
%     describe_location,
%     maybe_pickup_orb(player),
%     check_encounter,
%     check_exit_condition.
after_player_move :-
    describe_location,
    maybe_pickup_orb(player),
    check_exit_condition,                
    (   game_over(_) -> true             
    ;   check_encounter
    ).


maybe_pickup_orb(Agent) :-
    ( Agent = player -> here(Pos)
    ; Agent = enemy  -> enemy_here(Pos)
    ),
    ( orb_at(Pos) ->
        retract(orb_at(Pos)),
        retract(has_orb(_)),
        asserta(has_orb(Agent)),
        ( Agent = player -> asserta(owns(player, time_orb)) ; true ),
        format('~w take Time Orb! ~n', [Agent]),
        update_adversary_mode
    ; true).

check_exit_condition :-
    here(P), enemy_here(E),
    ( exit_at(P), has_orb(player) ->
        write('You reach the exit with the Time Orb and escape successfully!'), nl,
        asserta(game_over(win))
    ; exit_at(E), has_orb(enemy) ->
        write('The enemy escaped with the Time Orb. You have failed.'), nl,
        asserta(game_over(lose))
    ; true).

update_adversary_mode :-
    ( has_orb(player) ->
        retractall(adversary_mode(_)), asserta(adversary_mode(chaser)),
        write('The enemy has detected that you''ve obtained the Time Orb and is now pursuing you!'), nl
    ; has_orb(enemy) ->
        retractall(adversary_mode(_)), asserta(adversary_mode(thief)),
        write('The enemy has obtained the Time Orb and will attempt to reach the exit!'), nl
    ; true).

/* ===================== Debug Helpers ===================== */

debug_show_enemy_pos :-
    ( enemy_here(E) ->
        format('【DEBUG】 Enemy current position: ~w~n', [E])
    ;   write('【DEBUG】 Enemy position unknown (no enemy_here/1).'), nl
    ).

/* ===================== Enemy Turn & Planner Interaction ===================== */

enemy_turn :-
    % Do not act if the game already ended
    ( game_over(_) -> true
    ; enemy_here(OldE),
      choose_enemy_action(Action),
      ( apply_enemy_action(Action) ->
            format('The enemy is moving in the shadows.'),nl
      ;   % If planner produced an invalid action, fallback to random
          write('The planner''s action is invalid; the enemy will now move randomly.'), nl,
          fallback_random_enemy_move
      ),
      % After enemy movement: possible orb pickup / encounter / exit
      maybe_pickup_orb(enemy),
      check_exit_condition,              
      (   game_over(_) -> true           
      ;   check_encounter
      ),
      enemy_here(NewE),
      (OldE =\= NewE -> true ; true)
    ).

% % debug
% enemy_turn :-
%     ( game_over(_) -> true
%     ; enemy_here(OldE),
%       fallback_random_enemy_move,
%       maybe_pickup_orb(enemy),
%       check_encounter,
%       check_exit_condition,
%       enemy_here(NewE),
%       (OldE =\= NewE -> true ; true)
%     ).

/* Enemy action selection:
   Priority: use current plan → replan → fallback random
*/

choose_enemy_action(Action) :-
    current_plan([First|Rest]),
    !,
    ( valid_planned_action(First) ->
        Action = First,
        retractall(current_plan(_)),
        asserta(current_plan(Rest))
    ;   % First action invalid → replan
        recompute_plan,
        choose_enemy_action(Action)
    ).
choose_enemy_action(Action) :-
    % No current plan → replan
    recompute_plan,
    ( current_plan([First|Rest]) ->
        Action = First,
        retractall(current_plan(_)),
        asserta(current_plan(Rest))
    ; % planner no solution → random move
      Action = random_move
    ).

/* Simple static validity check for planner actions (here only move(X,Y)) */
% valid_planned_action(move(From,To)) :-
%     enemy_here(From),
%     valid_step(From, To), !.
valid_planned_action(move(From,To)) :-
    integer(From),
    integer(To),
    enemy_here(From),
    valid_step(From, To),
    !.
valid_planned_action(_) :- fail.

/* Apply enemy action to world state */

apply_enemy_action(move(_From, To)) :-
    enemy_here(Current),
    retract(enemy_here(Current)),
    asserta(enemy_here(To)),
    true.
apply_enemy_action(random_move) :-
    fallback_random_enemy_move.

/* Fallback random movement when no planner solution or error */

fallback_random_enemy_move :-
    enemy_here(E),
    (   % If the item has already “predicted” the next move, proceed according to the prediction.
        forced_enemy_move(To)
    ->  retract(forced_enemy_move(To)),
        retract(enemy_here(E)),
        asserta(enemy_here(To))
    ;   % Otherwise, random select a location.
        findall(Next,
                ( member(Dir,[north,south,east,west]),
                  compute_new_pos(E,Dir,Next),
                  valid_step(E,Next)
                ),
                Moves),
        (   Moves = []
        ->  true
        ;   random_member(Next,Moves),
            retract(enemy_here(E)),
            asserta(enemy_here(Next))
        )
    ).

/* Decide (once) a random next move for the enemy and store it in forced_enemy_move/1. */

ensure_forced_enemy_move(To) :-
    (   forced_enemy_move(To)
    ->  true                         % cache is already available, then use it directly.
    ;   enemy_here(E),
        findall(Next,
                ( member(Dir, [north,south,east,west]),
                  compute_new_pos(E, Dir, Next),
                  valid_step(E, Next)
                ),
                Moves),
        (   Moves = []
        ->  To = E                   % can't move, then just stay
        ;   random_member(To, Moves) % randomly select a target location
        ),
        asserta(forced_enemy_move(To))
    ).

/* Predict enemy next move: used by map_fragment item */

predict_enemy_next(Info) :-
    (   current_plan([First|_]),
        First \= random_move
    ->  format(atom(Info), 'The next step is to ~w', [First])
    %   if no planner, then use ensure_forced_enemy_move/1
    ;   ensure_forced_enemy_move(To),
        enemy_here(E),
        format(atom(Info),
               'The enemy will move from ~w to ~w',
               [E, To])
    ).

/* ===================== Planning.Domains Interaction ===================== */

recompute_plan :-
    % Generate dynamic problem file based on the current game state
    generate_dynamic_problem('adversary_problem_dynamic.pddl'),
    % Call Planning.Domains online solver
    call_planner('adversary_domain.pddl',
                 'adversary_problem_dynamic.pddl',
                 PlanAtoms),
    retractall(current_plan(_)),
    asserta(current_plan(PlanAtoms)).

/* Write current state into a dynamic PDDL problem file */

generate_dynamic_problem(Filename) :-
    open(Filename, write, Stream),
    here(PlayerPos),
    enemy_here(EnemyPos),
    ( orb_at(OrbPos) -> true ; OrbPos = none ),
    exit_at(ExitPos),
    % Mode switch: thief → reach exit with orb; chaser → reach player's location
    adversary_mode(Mode),
    write_problem_pddl(Stream, Mode, PlayerPos, EnemyPos, OrbPos, ExitPos),
    close(Stream).

write_problem_pddl(Stream, Mode, PlayerPos, EnemyPos, OrbPos, ExitPos) :-
    % Location constants loc-1 ... loc-16
    format(Stream,'(define (problem time-orb-problem)~n', []),
    format(Stream,'  (:domain time-orb-adversary)~n', []),
    format(Stream,'  (:objects~n', []),
    forall(between(1,16,I),
           format(Stream,'    loc-~d - loc~n',[I])),
    format(Stream,'  )~n', []),

    format(Stream,'  (:init~n', []),
    format(Stream,'    (at-enemy loc-~d)~n',[EnemyPos]),
    format(Stream,'    (player-at loc-~d)~n',[PlayerPos]),
    ( OrbPos \= none ->
        format(Stream,'    (orb-at loc-~d)~n',[OrbPos])
    ; true),
    format(Stream,'    (exit-at loc-~d)~n',[ExitPos]),
    % Write adjacency relations
    write_adjacencies(Stream),
    format(Stream,'  )~n', []),

    % Goal
    format(Stream,'  (:goal~n    ', []),
    ( Mode = thief ->
        format(Stream,'(escaped)~n', [])
    ; Mode = chaser ->
        % Enemy reaches player's location
        format(Stream,'(and (at-enemy loc-~d) (player-at loc-~d))~n',[PlayerPos,PlayerPos])
    ; % Default thief mode
      format(Stream,'(escaped)~n', [])
    ),
    format(Stream,'  )~n)~n', []).

write_adjacencies(Stream) :-
    forall(between(1,16,From),
      forall(between(1,16,To),
        ( adjacent(From,To) ->
            format(Stream,'    (adj loc-~d loc-~d)~n',[From,To])
        ; true))).

/* 4x4 grid adjacency: up/down/left/right */
adjacent(A,B) :-
    RowA is (A - 1) // 4,
    ColA is (A - 1) mod 4,
    RowB is (B - 1) // 4,
    ColB is (B - 1) mod 4,
    ( (RowA =:= RowB, abs(ColA-ColB) =:= 1)
    ; (ColA =:= ColB, abs(RowA-RowB) =:= 1)
    ).

/* Call Planning.Domains online solver and parse JSON response */

call_planner_inner(DomainFile, ProblemFile, PlanAtoms) :-
    % writeln('DEBUG: call_planner ENTERED'),
    % format('DEBUG DomainFile = ~w, ProblemFile = ~w~n', [DomainFile, ProblemFile]),

    URL = 'http://solver.planning.domains/solve',
    catch(
        (
            read_file_to_string(DomainFile, DomainStr, []),
            read_file_to_string(ProblemFile, ProblemStr, []),

            Payload = _{domain:DomainStr, problem:ProblemStr},

            setup_call_cleanup(
                http_post(URL,
                          json(Payload),
                          ReplyStream,
                          [ timeout(20) ]),
                json_read_dict(ReplyStream, Dict),
                close(ReplyStream)
            ),

            (   Dict.get(status) = "ok"
            ->  Result   = Dict.get(result),
                PlanList = Result.get(plan),
                % format('DEBUG: Raw PlanList = ~q~n', [PlanList]),
                maplist(step_name_to_atom, PlanList, PlanAtoms)
                % format('DEBUG: Parsed PlanAtoms = ~q~n', [PlanAtoms])
            ;   write('Planning.Domains returned error or no solution.'), nl,
                PlanAtoms = []
            )
        ),
        Error,
        (
            % print_message(error, Error),
            % write('Planning.Domains unreachable or PDDL read error; enemy will move randomly.'), nl,
            PlanAtoms = []
        )
    ).

call_planner(DomainFile, ProblemFile, PlanAtoms) :-
    (   call_planner_inner(DomainFile, ProblemFile, PlanAtoms0)
    ->  PlanAtoms = PlanAtoms0
    ;   % if inner failed
        % writeln('DEBUG: call_planner_inner FAILED, using empty plan.'),
        PlanAtoms = []
    ).

normalize_loc(LocTerm, N) :-
    LocTerm =.. ['-', loc, N],  
    integer(N),
    !.
normalize_loc(N, N).

step_name_to_atom(StepDict, AtomName) :-
    % format('DEBUG: step_name_to_atom input StepDict = ~q~n', [StepDict]),
    (   get_dict(name, StepDict, NameStr)
    ->  atom_string(NameAtom, NameStr),
        % example: NameStr = "(move loc-16 loc-12)"
        atom_to_term(NameAtom, Term, _),
        % format('DEBUG: parsed Term from name = ~q~n', [Term]),
        (   Term =.. [Functor|RawArgs]
        ->  maplist(normalize_loc, RawArgs, Args),
            AtomName =.. [Functor|Args]
        ;  
            AtomName = Term
        )
    ;   % if name field doesn't exist
        % format('DEBUG WARNING: no name key in StepDict: ~q~n', [StepDict]),
        AtomName = noop
    ).



/* ===================== Encounter, Robbery & Scuffle ===================== */

check_encounter :-
    game_over(_),      
    !.
check_encounter :-
    here(P),
    enemy_here(E),
    ( P =:= E ->
        handle_encounter
    ; true).

handle_encounter :-
    cooldown(CD),
    ( (has_orb(player); has_orb(enemy)), CD =:= 0 ->
        % Someone has the orb & no cooldown → robbery
        orb_robbery
    ;   % otherwise non-lethal scuffle
        encounter_no_orb
    ).

/* Robbery:
   Someone holds the orb → random chance to steal or defend
   Then both are scattered randomly and enter cooldown
*/
orb_robbery :-
    has_orb(Who),
    Who \= none,
    !,
    format('You and your enemy meet in the same square, erupting into a fierce battle over the Time Orb! ~n', []),
    random_between(0,1,R),
    ( R =:= 0 ->
        % Current holder keeps it
        format('~w fiercely guarded the Time Orb. ~n', [Who]),
        NewOwner = Who
    ;   % Other side steals it
        ( Who = player -> NewOwner = enemy ; NewOwner = player ),
        format('~w snatch the Time Orb!~n', [NewOwner])
    ),
    retractall(has_orb(_)),
    asserta(has_orb(NewOwner)),
    ( NewOwner = player ->
        ( owns(player,time_orb) -> true ; asserta(owns(player,time_orb)) )
    ;   % Enemy holds orb; remove from player if necessary
        ( retract(owns(player,time_orb)) -> true ; true )
    ),
    scatter_after_robbery,
    % Set cooldown: 3 rounds
    retract(cooldown(_)),
    asserta(cooldown(3)),
    update_adversary_mode.

orb_robbery :-
    % This clause should not occur (handled above)
    encounter_no_orb.

/* Random scattering after robbery:
   Must not land on same cell or on exit
*/
scatter_after_robbery :-
    exit_at(Exit),
    findall(Pos, between(1,16,Pos), All),
    exclude(=(Exit), All, Candidates),
    random_member(PPos, Candidates),
    exclude(=(PPos), Candidates, Rest),
    random_member(EPos, Rest),
    retract(here(_)), asserta(here(PPos)),
    retract(enemy_here(_)), asserta(enemy_here(EPos)),
    format('After the scramble ends, a wave of dizziness sweeps over you as you and your enemy are flung to two random locations... ~nYou can use the map to find out where you are.', []).

/* Non-lethal scuffle:
   Use history_state/2 to push each other back
   If no history exists → random scatter
*/
encounter_no_orb :-
    write('You and your enemy are locked in a fierce struggle within the cramped confines of the room (though not to the point of death).'),
    ( history_state(PPrev,EPrev) ->
        random_between(0,1,R),
        ( R =:= 0 ->
            % Enemy pushes player
            retract(here(_)), asserta(here(PPrev)),
            format('The enemy pushed you back into the room you were in before (~w). ~n',[PPrev])
        ;   % Player pushes enemy
            retract(enemy_here(_)), asserta(enemy_here(EPrev)),
            format('You shoved him hard, driving the enemy back into the previous room (~w). ~n',[EPrev])
        )
    ;   % No history -> random scatter
        write('You''ve all forgotten where you were just now, feeling only the world spinning around you...~n'),
        scatter_both_randomly
    ).

scatter_both_randomly :-
    exit_at(Exit),
    findall(Pos, between(1,16,Pos), All),
    exclude(=(Exit), All, Candidates),
    random_member(PPos, Candidates),
    exclude(=(PPos), Candidates, Rest),
    random_member(EPos, Rest),
    retract(here(_)), asserta(here(PPos)),
    retract(enemy_here(_)), asserta(enemy_here(EPos)),
    write('Both parties were randomly scattered to two different locations on the map. ~n').

/* ===================== End Conditions ===================== */

:- dynamic game_over/1.  % game_over(win) / game_over(lose)

check_end :-
    ( game_over(win) ->
        write('=== Congratulations, you won! ==='), nl, !, retractall(game_over(_))
    ; game_over(lose) ->
        write('=== Sorry, but you lost. ==='), nl, !, retractall(game_over(_))
    ; fail).


