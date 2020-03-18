function  [Acc,RMSE] = Score_acc_Calculate(RT_EST, Pair)
theta = 0.65;
delta_score = RT_EST(Pair(:,1)) - RT_EST(Pair(:,2));
flag = 1 ./ (1 + exp(-delta_score));
Idx_flag = flag;
flag(Idx_flag > theta) = 1;
flag(Idx_flag < theta) = 0;
flag(Idx_flag < 1 - theta) = -1;

Acc = (sum(flag == 1) + 0.5 * sum(flag == 0)) / size(flag, 1);

Score = zeros(size(RT_EST,1), 2);
for n = 1: length(flag)
    temp_score = zeros(1, 2);
    temp_score(1) = 1;
    temp_score(2) = flag(n);
    
    Score(Pair(n, 1), :) = Score(Pair(n, 1), :) + (temp_score == 1) + 0.5 * (temp_score == 0);
    Score(Pair(n, 2), :) = Score(Pair(n, 2), :) + (temp_score == -1) + 0.5 * (temp_score == 0); 
end
RMSE = sqrt(mean((Score(:, 1) - Score(:, 2)).^ 2));