%% success rate

% anymal
%t 0
%g 57
%c 43
%o 0

cnn_file = 'cnn.csv';
lstm_file = 'lstm.csv';
tcn_file1 = 'tcn1.csv';
tcn_file2 = 'tcn2.csv';

cnn_t = readtable(cnn_file);
lstm_t = readtable(lstm_file);
tcn_t1 = readtable(tcn_file1);
tcn_t2 = readtable(tcn_file2);

figure('Position',[0 0 800 600])
plot(cnn_t.stage, cnn_t.successRate, 'r', 'Linewidth', 2)
hold on
plot(lstm_t.stage, lstm_t.successRate, 'g', 'Linewidth', 2)
plot(tcn_t1.stage, tcn_t1.successRate, 'c', 'Linewidth', 2)
plot(tcn_t2.stage, tcn_t2.successRate, 'b', 'Linewidth', 2)
hold off
ax1 = gca;  % current axes
ax1.FontSize = 20;

title('Success Rate in Occupancy Rate = 1')
xlabel('Curriculum Stage')
ylabel('Goal/(100 trial)')
ylim([0, 1.0])
legend('CNN policy', 'LSTM policy', 'TCN policy (2 TCN layer)', 'location', 'southoutside')

%% learning curve

cnn_file = 'aecnn-5hz.csv';
lstm_file = 'aelstm-5hz.csv';
tcn_file1 = 'aetcn-1layer-5hz.csv';
tcn_file2 = 'aetcn-2layer-5hz.csv';

cnn_t = readtable(cnn_file);
lstm_t = readtable(lstm_file);
tcn_t1 = readtable(tcn_file1);
tcn_t2 = readtable(tcn_file2);

max_timestep = 2190;

cnn_eprew = cnn_t.eprew(1:max_timestep);
lstm_eprew = lstm_t.eprew(1:max_timestep);
tcn_eprew1 = tcn_t1.eprew(1:max_timestep);
tcn_eprew2 = tcn_t2.eprew(1:max_timestep);

step = 1:max_timestep;
[cnn_eprew, cnn_mask] = rmoutliers(cnn_eprew);
[lstm_eprew, lstm_mask] = rmoutliers(lstm_eprew);
[tcn_eprew1, tcn_mask1] = rmoutliers(tcn_eprew1);
[tcn_eprew2, tcn_mask2] = rmoutliers(tcn_eprew2);

cnn_step = step(~cnn_mask);
lstm_step = step(~lstm_mask);
tcn_step1 = step(~tcn_mask1);
tcn_step2 = step(~tcn_mask2);

cnn_eprew = tsmovavg(cnn_eprew, 's', 3, 1);
lstm_eprew = tsmovavg(lstm_eprew, 's', 3, 1);
tcn_eprew1 = tsmovavg(tcn_eprew1, 's', 3, 1);
tcn_eprew2 = tsmovavg(tcn_eprew2, 's', 3, 1);

figure('Position',[0 0 800 600])
plot(cnn_step, cnn_eprew, 'r', 'Linewidth', 2)
hold on
plot(lstm_step, lstm_eprew, 'g', 'Linewidth', 2)
plot(tcn_step1, tcn_eprew1, 'b', 'Linewidth', 2)
plot(tcn_step2, tcn_eprew2, 'b', 'Linewidth', 2)
hold off
ax1 = gca;  % current axes
ax1.FontSize = 20;

title('Episode Mean Reward')
xlim([0, 2200])
xlabel('Iteration')
ylabel('Episode Mean Reward')
legend('CNN policy', 'LSTM policy', 'TCN policy (2 TCN layer)', 'location', 'southoutside')

%% Occupancy 

full_occupancy = 0.8;
stage = [0];
occupancy = [0];

for i = 0:10
    stage(end+1) = i + 1;
    occupancy(end+1) = 0.3 ^ (0.7 ^ i) * full_occupancy;
end

step = 0:199;
lr = 5e-5 * ones(1, 200);
lr_start = 5e-5;
lr_end = 1e-5;
power = 2;

for i = 200:2199
    step(end+1) = i;
    lr(end+1) = (lr_start - lr_end) * (1 - (i - 200) / (2199 - 200)) ^ power + lr_end;
end

figure('Position',[0 0 1200 900])
ax1 = gca;  % current axes
ax1.FontSize = 20;
ax1.XColor = 'r';
ax1.YColor = 'r';
ax1.XLabel.String = 'Curriculum Stage';
ax1.XLabel.Color = 'k';
ax1.YLabel.String = 'Occupancy Rate';
ax1.YLabel.Color = 'k';
ax1.XLim = [0, 11];
line(stage, occupancy, 'Color', 'r', 'Linewidth', 2)
%xlabel('Curriculum Stage')
%ylabel('Occupancy Rate')

ax2 = axes('Position',ax1.Position,...
  'XAxisLocation','top',...
  'YAxisLocation','right',...
  'Color','none',...
  'XColor','b',...
  'YColor','b');
ax2.FontSize = 20;
ax2.XLabel.String = 'Iteration';
ax2.XLabel.Color = 'k';
ax2.YLabel.String = 'Learning Rate';
ax2.YLabel.Color = 'k';
ax2.XLim = [0, 2200];
line(step, lr, 'Linewidth', 2, 'Color', 'b', 'Parent', ax2)
