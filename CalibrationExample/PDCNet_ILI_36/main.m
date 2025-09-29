clc;clear all;close all;
%% Basic data import
window = 3;
true = readNPY('true.npy');
A = readNPY('pred.npy');
B = A;
memorynum = 16;% number of memories
W = size(A,2);
Runtimes = size(A,1) - W - memorynum + 1;% Total runs
threshold = 0.23;
v = 0.3;
%% 
p = 0;
while p < Runtimes
%% Calculate the coefficient and get the correction coefficient dictionary
% retrieval memory
memory = A(p+1:p+memorynum,:); %Note the location of memory and current predictions, there is no data breach
memory_true = true(p+1:p+memorynum,:);
mse = mean((memory(:) - memory_true(:)).^2);
if threshold <=  mse
    p = p + 1;  
    continue;  %If the threshold condition does not pass, the correction is abandoned
end 
% Regression analysis is performed on each window
i = 1;
j = 1;
[h, l] = size(memory);
regression = memory;
while i < h+1
    while j*window < l+1 
        regression(i,j*window+1-window:j*window) = polyval(polyfit(1:window, memory(i,j*window+1-window:j*window), 1), 1:1:window);
        j = j+1;
    end
    i = i+1;
    j = 1;
end
% Calculate abrupt factor M
x = mean(mean(memory));%Take the average value of memory as the unit of the x dimension
i = 1;
j = 1;
[h, l] = size(regression);
M = [];
while i < h+1
    while j*window < l+1 
        M(i,j) = abs((regression(i,j*window)-regression(i,j*window-1))/x);
        j = j+1;
    end
    i = i+1;
    j =1 ;
end
% Cluster, divide M into ? categories
data = M(:);  
dbscan_idx = dbscan(data, 0.002, 5);
numClusters = max(dbscan_idx);
%%
[idx, C] = kmeans(data, numClusters);
idx = reshape(idx, size(M));
C = C';
% Calculated correction factor
CorrectionFactor = [];
k = 1;
i = 1;
j = 1;
sum_pre = 0;
sum_true = 0;
while k < (numClusters + 1)
  while i < h+1
    while j*window < l+1 
        if idx(i,j) == k
            sum_pre = sum_pre + sum(memory(i,j*window-window+1:j*window));
            sum_true = sum_true + sum(memory_true(i,j*window-window+1:j*window));
        end
        j = j+1;
    end
    i = i+1;
    j =1 ;
  end
  CorrectionFactor(k) = sum_true / sum_pre;
  sum_pre = 0;
  sum_true = 0;
  i = 1;
  j = 1;
  k = k + 1;
end
% Computed to get the correction dictionary
dictionary = [];
dictionary(1,:) = C; dictionary(2,:) = CorrectionFactor;
dictionary(2, dictionary(2, :) > 1+v | dictionary(2, :) < 1-v) = 1;%Normalize outliers
%% Start correction
%Correction object
tar = B(p + W + memorynum , :);  %Note the location of memory and current predictions, there is no data breach
% tar_regression
j = 1;
w = size(tar,2);
tar_regression = tar;
while j*window < w+1 
    tar_regression(j*window+1-window:j*window) = polyval(polyfit(1:window, tar(j*window+1-window:j*window), 1), 1:1:window);
    j = j+1;
end
% M Factor
j = 1;
tar_M = [];
while j*window < w+1 
    tar_M(j) = abs((tar_regression(j*window)-tar_regression(j*window-1))/x);%Classify tar_M
    j = j+1;
end
% Start correction
j = 1;
tar_correction = tar;
while j*window < w+1 
    [~, index] = min(abs(dictionary(1, :) - tar_M(j)));
    xishu = dictionary(2, index);
    tar_correction(j*window-window+1:j*window) = tar(j*window-window+1:j*window) .* xishu;
    j = j+1;
end
% A  new correction result.Save in B
B(p + W + memorynum , :) = tar_correction;
p = p + 1;
end
%%
filename = 'correction.npy'; 
writeNPY(B, filename); 

%% Visual display
true = readNPY('true.npy');
FirstForecast = readNPY('pred.npy');
SecondForecast = readNPY('correction.npy');
S = 69;
true = true(S,:);
FirstForecast = FirstForecast(S,:);
SecondForecast = SecondForecast(S,:);
  
Timestep = 1:36;  

figure;   
hold on; 
plot(Timestep, FirstForecast, 'b-', 'DisplayName', 'Prediction');  
plot(Timestep, SecondForecast, 'r-', 'DisplayName', 'WithCalibration');  
plot(Timestep, true, 'k-', 'DisplayName', 'GroundTruth');   
 
legend('show');  
xlabel('Timestep'); 
ylabel('Value');   

grid on;   
hold off;  

%% Test correction effect

true = readNPY('true.npy');
A = readNPY('pred.npy');
B = readNPY('correction.npy');
mse_A = mean((A - true).^2, 2);  
mse_B = mean((B - true).^2, 2);  

improvement = ((mse_A - mse_B) ./ mse_A) * 100; 

figure;  
plot(improvement, '-o'); 
xlabel('Row Index');  
title('PDCNet ILI'); 
ylabel('Improvement(%)');  
grid on; 
%%
true = readNPY('true.npy');
pred = readNPY('pred.npy');
repred = readNPY('correction.npy');

% Identify the rows that are different between pred and repred
diff_rows = any(pred ~= repred, 2);

% Extract the rows with differences
true_diff = true(diff_rows, :);
pred_diff = pred(diff_rows, :);
repred_diff = repred(diff_rows, :);

% The roughly predicted MSE/MAE at the correction position
mse_pred = mean(mean((true_diff - pred_diff).^2, 2));
mae_pred = mean(mean(abs(true_diff - pred_diff), 2));

% The corrected MSE/MAE at the correction position
mse_repred = mean(mean((true_diff - repred_diff).^2, 2));
mae_repred = mean(mean(abs(true_diff - repred_diff), 2));

% Show the result
fprintf('The roughly predicted MSE at the correction position: %.4f\n', mse_pred);
fprintf('The roughly predicted MAE at the correction position: %.4f\n', mae_pred);
fprintf('The corrected MSE at the correction position: %.4f\n', mse_repred);
fprintf('The corrected MAE at the correction position: %.4f\n', mae_repred);
