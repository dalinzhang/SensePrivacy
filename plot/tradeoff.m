clear all;

% motion sense dataset
style_weight = [0.05, 0.25, 0.45, 0.55, 0.65, 0.85, 0.95];

% activity_acc = [0.9360, 0.9117, 0.9100, 0.9225, 0.9093, 0.8948, 0.7988]*100;
% gender_acc   = [0.8107, 0.5980, 0.5375, 0.5679, 0.5421, 0.5658, 0.5785]*100;
% id_acc       = [0.5790, 0.1363, 0.0814, 0.0751, 0.0657, 0.0558, 0.0300]*100;

% height_error = [56.6461, 688.3771, 935.6166,  1011.6985, 1388.5634, 1825.1231, 3084.9627]/48.0295;
% weight_error = [144.5316, 2139.5907, 3116.9433, 3966.9314, 5099.9164, 5550.8222, 9570.2185]/84.2879;
% age_error    = [20.0515, 64.7458, 65.2666,   115.82470, 74.8300,   65.9482,   149.9757]/14.6403;

% height_error = [56.6461, 688.3771, 935.6166,  1011.6985, 1388.5634, 1825.1231, 3084.9627];
% weight_error = [144.5316, 2139.5907, 3116.9433, 3966.9314, 5099.9164, 5550.8222, 9570.2185];
% age_error    = [20.0515, 64.7458, 65.2666,   115.82470, 74.8300,   65.9482,   149.9757];


% mobiact sense dataset
activity_acc = [0.9563, 0.9548, 0.9363, 0.9222, 0.8585, 0.7057, 0.2056]*100;
gender_acc   = [0.7694, 0.8247, 0.7053, 0.6926, 0.6588, 0.6799, 0.6818]*100;
id_acc       = [0.7310, 0.6290, 0.3000, 0.1970, 0.0831, 0.0234, 0.02188]*100;

height_error = [65.1452, 146.5535, 567.5358,  1052.6009, 1187.54, 1283.58125, 2335.5746];
weight_error = [312.6940, 340.8019, 2614.3292, 3672.1963, 4250.489, 6786.609, 15178.1847];
age_error    = [11.0247, 13.6693,    36.8400,   38.8297, 63.6931, 94.6428, 104.8561];

% data = transpose([activity_acc; gender_acc; id_acc]);
% yyaxis left
% h = bar(style_weight, data, 'EdgeAlpha',0, 'BarWidth', 1);
% h(1).FaceColor = [79 195 247]/255;
% h(2).FaceColor = [174 213 129]/255;
% h(3).FaceColor = [186 104 200]/255;

yyaxis left
act_plot 	= plot(style_weight, activity_acc, '-*', 'MarkerSize',15, 'DisplayName', 'Activity','LineWidth',3, 'Color',[79 195 247]/255);
hold on;
gen_plot 	= plot(style_weight, gender_acc, '-s', 'MarkerSize',15, 'DisplayName', 'Gender','LineWidth',3, 'Color', [174 213 129]/255);
hold on;
id_plot 	= plot(style_weight, id_acc, '-d', 'MarkerSize',15, 'DisplayName', 'ID','LineWidth',3, 'Color', [186 104 200]/255);
hold on;

ylabel('Accuracy','FontSize',90,'FontWeight','bold');


a = get(gca,'TickLabel');
yyaxis right
height_plot 	= plot(style_weight, height_error, '-x', 'MarkerSize',15, 'DisplayName', 'Height','LineWidth',3, 'Color', [255 193 7]/255);
hold on;
weight_plot 	= plot(style_weight, weight_error, '-^', 'MarkerSize',15, 'DisplayName', 'weight','LineWidth',3, 'Color', [229 115 115]/255);
hold on;
age_plot 	    = plot(style_weight, age_error, '-o', 'MarkerSize',15, 'DisplayName', 'Age','LineWidth',3, 'Color', [255 87 34]/255);
hold on;

ylabel('Mean Squared Error (MSE)','FontSize',80,'FontWeight','bold');
set(gca,  'YScale', 'log','TickLabel', a, 'fontsize',25,'FontWeight','bold','linewidth',2, 'Color', 'r'); 

xlim([0 1])
% set(gca, 'XTick', (0:0.5:2)); 

xlabel('Stlye Loss Weight','FontSize',40,'FontWeight','bold');

% set(gca, 'XTickLabel',str, 'XTick',1:numel(str))
set(gca, 'FontSize',35,'FontWeight','bold','linewidth',2); % set axes style
set(gca,'Color',[0.95 0.95 0.95]);
set(gca,'TickLength',[0 0]);
ax = gca;
ax.GridColor = [1,1,1];
ax.GridAlpha = 1;

grid on

% set legend
lgd =legend([]);
lgd.Box = 'off';

box off
lgd.LineWidth = 1.5;
lgd.FontSize = 35;
lgd.FontWeight = 'bold';

hold off;
