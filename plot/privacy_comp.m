clear all;

str = {'activity'; 'gender'; 'ID'; 'height'; 'weight'; 'age'};

train_proportion = [1,2,3,4,5,6];
% overview motionsense
% raw   = [0.9349*100, 0.9505*100, 0.7264*100, 0, 0, 0];
% trans = [0.9225*100, 0.5679*100, 0.0751*100, 0, 0, 0];
% 
% raw_e   = [0, 0, 0, 48.0294, 84.2878, 14.6402];
% trans_e = [0, 0, 0, 1011.6985, 3966.9314, 115.8247];

% overview mobiact
% raw   = [0.9617*100, 0.8920*100, 0.8130*100, 0, 0, 0];
% trans = [0.9222*100, 0.6926*100, 0.1970*100, 0, 0, 0];
% 
% raw_e   = [0, 0, 0, 53.5790, 164.2553, 5.8570];
% trans_e = [0, 0, 0, 1052.6009, 3672.1963, 38.8297];


% activity loss only
% raw	 = [0.9349*100, 0.9505*100, 0.7264*100, 0, 0, 0];
% trans= [0.9283*100, 0.8754*100, 0.5422*100, 0, 0, 0];
% 
% raw_e	 = [0, 0, 0, 48.0294, 84.2878, 14.6402];
% trans_e  = [0, 0, 0, 61.2791, 141.6211, 25.5209];

% % style loss only
% raw	 = [0.9349*100, 0.9505*100, 0.7264*100, 0, 0, 0];
% trans= [0.2039*100, 0.5706*100, 0.0331*100, 0, 0, 0];
% 
% raw_e	 = [0, 0, 0, 48.0294, 84.2878, 14.6402];
% trans_e  = [0, 0, 0, 2830.6368, 2064.3524, 150.4011];

% % content loss only
% raw	 = [0.9349*100, 0.9505*100, 0.7264*100, 0, 0, 0];
% trans= [0.9250*100, 0.9028*100, 0.6154*100, 0, 0, 0];
% 
% raw_e	 = [0, 0, 0, 48.0294, 84.2878, 14.6402];
% trans_e  = [0, 0, 0, 46.2136, 116.9093, 20.1313];

% with reconstruction loss
raw	 = [0.9349*100, 0.9505*100, 0.7264*100, 0, 0, 0];
trans = [0.9225*100, 0.5679*100, 0.0751*100, 0, 0, 0];
trans_recons = [0.8832*100, 0.5815*100, 0.1395*100, 0, 0, 0];

raw_e	 = [0, 0, 0, 48.0294, 84.2878, 14.6402];
trans_e  = [0, 0, 0, 1011.6985, 3966.9314, 115.8247];
trans_recons_e = [0, 0, 0, 113.3794, 1171.8974, 328.6979];

data = transpose([raw; trans]);
data_e = transpose([raw_e; trans_e]);

% data = transpose([raw; trans; trans_recons]);
% data_e = transpose([raw_e; trans_e; trans_recons_e]);

yyaxis left
h = bar(train_proportion, data, 'EdgeAlpha',0, 'BarWidth', 1);% EdgeAlpha sets the box line of the bar; BarWidth sets the width of the bar
h(1).FaceColor = [79 195 247]/255;
h(2).FaceColor = [174 213 129]/255;
% h(3).FaceColor = [186 104 200]/255;
ylim([0 100])
ylabel('Accuracy','FontSize',40,'FontWeight','bold');

hold on

a = get(gca,'TickLabel');
yyaxis right
h_e = bar(train_proportion, data_e, 'EdgeAlpha',0, 'BarWidth', 1);% EdgeAlpha sets the box line of the bar; BarWidth sets the width of the bar

h_e(1).FaceColor = [229 115 115]/255;
h_e(2).FaceColor = [255 238 88]/255;
% h_e(3).FaceColor = [255 87 34]/255;

% ylim([10 10000])
ylabel('Mean Squared Error (MSE)','FontSize',30,'FontWeight','bold');
set(gca,  'YScale', 'log','TickLabel', a, 'fontsize',25,'FontWeight','bold','linewidth',2); 

set(gca, 'XTickLabel',str, 'XTick',1:numel(str))
set(gca, 'FontSize',30,'FontWeight','bold','linewidth',2); % set axes style
set(gca,'Color',[0.95 0.95 0.95]);
set(gca,'TickLength',[0 0]);
ax = gca
ax.GridColor = [1,1,1];
ax.GridAlpha = 1
% color of the bar chart
% colormap(summer(9)); 

% set the legend
l = cell(1,3);
l{1}='raw data'; l{2}='transformed data';  l{3}='transformed data with reconstruction loss';
lgd = legend(h, l, 'Box', 'off');
lgd.FontSize=25;

xlabel('Test Information','FontSize',30,'FontWeight','bold');

box off
grid on