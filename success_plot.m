%% success rate

cnn_file = 'cnn.csv';
lstm_file = 'lstm.csv';
tcn_file = 'tcn.csv';

cnn_t = readtable(cnn_file);
lstm_t = readtable(lstm_file);
tcn_t = readtable(tcn_file);

figure('Position',[0 0 600 600])
plot(cnn_t.stage, cnn_t.successRate, 'r', 'Linewidth', 2)
hold on
plot(lstm_t.stage, lstm_t.successRate, 'g', 'Linewidth', 2)
plot(tcn_t.stage, tcn_t.successRate, 'b', 'Linewidth', 2)
hold off

title('Success Rate in Occupancy Rate = 1')
xlabel('Curriculum Stage')
ylabel('Goal/(100 trial)')
ylim([0, 1.0])
legend('CNN policy', 'LSTM policy', 'TCN policy (2 TCN layer)', 'location', 'southoutside')

%% learning curve

cnn_file = 'aecnn-5hz.csv';
lstm_file = 'aelstm-5hz.csv';
tcn_file = 'aetcn-2layer-5hz.csv';

cnn_t = readtable(cnn_file);
lstm_t = readtable(lstm_file);
tcn_t = readtable(tcn_file);

cnn_t = readtable(cnn_file);
lstm_t = readtable(lstm_file);
tcn_t = readtable(tcn_file);

cnn_eprew = cnn_t.eprew(1:2190);
lstm_eprew = lstm_t.eprew(1:2190);
tcn_eprew = tcn_t.eprew(1:2190);

% TODO time step should be remained!
cnn_eprew = rmoutliers(cnn_eprew);
lstm_eprew = rmoutliers(lstm_eprew);
tcn_eprew = rmoutliers(tcn_eprew);

cnn_eprew = tsmovavg(cnn_eprew, 's', 2, 1);
lstm_eprew = tsmovavg(lstm_eprew, 's', 2, 1);
tcn_eprew = tsmovavg(tcn_eprew, 's', 2, 1);

figure('Position',[0 0 600 600])
plot(cnn_eprew, 'r', 'Linewidth', 2)
hold on
plot(lstm_eprew, 'g', 'Linewidth', 2)
plot(tcn_eprew, 'b', 'Linewidth', 2)
hold off

title('Episode Mean Reward')
xlabel('Iteration')
ylabel('Episode Mean Reward')
legend('CNN policy', 'LSTM policy', 'TCN policy (2 TCN layer)', 'location', 'southoutside')

%% Occupancy 

for i = 0:
0.3 ^ (0.7 ^ i)

