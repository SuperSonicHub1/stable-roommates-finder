%*
Takes strong inspiration from doi:10.1017/S1471068425100240
*%

% Parameters
#const nagents = 4.
#const nmatchings = 1.

%% Objects
num_agents(nagents).
:- num_agents(N), N \ 2 != 0. % Even number of agents
:- num_agents(N), N < 2. % Number of agents >= 2
num_matchings(nmatchings).
agent(1..N) :- num_agents(N).
matching(1..K) :- num_matchings(K).

%% Directives
% Dedupe on ranks
#project rank/3.
% Show ranks and matchings
#show rank/3.
#show matched/3.

%% Ranking
% Everyone ranks themselves last
rank(A1, A1, N) :- agent(A1), num_agents(N).
% Unique ranks
:- rank(A1, A2, R1), rank(A1, A2, R2), R1 != R2, agent(A1), agent(A2).
% Consecutive ranks
:- rank(A, _, R), R > 1, not rank(A, _, R - 1), agent(A).
% Create lists
{ rank(A1, A2, R) : R=1..(N-1) } :- agent(A1), agent(A2), num_agents(N).
% Each person's preference list should be N people long
:- #count{A2 : agent(A2), rank(A1, A2, _) } != N, agent(A1), num_agents(N).

%% Matching
% Mutual acceptance
acceptable(A1, A2) :- rank(A1, A2, _), rank(A2, A1, _), A1 != A2, agent(A1), agent(A2).
% A prefers A1 to A2
prefers(A, A1, A2) :- rank(A, A1, R1), rank(A, A2, R2), R1 < R2, agent(A), agent(A1), agent(A2).
% Assignment
{ matched(A1, A2, I) : acceptable(A1, A2) }1 :- agent(A1), agent(A2), matching(I).
% Assignment symmetric
:- matched(A1, A2, I), not matched(A2, A1, I), agent(A1), agent(A2), matching(I).
% One-to-one
:- matched(A, A1, I), matched(A, A2, I), A1 != A2, agent(A), agent(A1), agent(A2), matching(I).
% Perfect matching (no singles)
:- not matched(A, _, I), agent(A), matching(I).
% Different pairs
has_different_pairs(I1, I2) :- matched(A, A1, I1), matched(A, A2, I2), A1 != A2, agent(A), agent(A1), agent(A2), matching(I1), matching(I2).
% Matchings must have at least one pair not in common
:- matching(I1), matching(I2), I1 != I2, not has_different_pairs(I1, I2).

%% Stability
% Blocking pairs
blocking_pair(A1, A2, I) :- 
	matched(A1, X, I), matched(A2, Y, I),
	prefers(A1, A2, X), prefers(A2, A1, Y),
	agent(A1), agent(A2), agent(X), agent(Y), matching(I).
% No blocking pairs
:- blocking_pair(A1, A2, I), agent(A1), agent(A2), matching(I).
