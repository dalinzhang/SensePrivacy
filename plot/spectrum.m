clear all;

test_data = load('/home/dadafly/program/SensePrivacy/motion-sense/data/test_data.mat');
trans_data = load('/home/dadafly/program/SensePrivacy/motion-sense/data/test_trans.mat');

% raw_data = test_data.test_data(320:320+100,:,:);
% new_data = trans_data.test_trans(320:320+100,:,:);

raw_data = test_data.test_data(1:86,:,:);
new_data = trans_data.test_trans(1:86,:,:);

raw_rotation_x = transpose(squeeze(raw_data(1,10,:)));
for i = 2:50
    raw_rotation_x = [raw_rotation_x transpose(squeeze(raw_data(i,10,:)))];
end


new_rotation_x = transpose(squeeze(new_data(1,10,:)));
for i = 2:50
    new_rotation_x = [new_rotation_x transpose(squeeze(new_data(i,10,:)))];
end


figure
spectrogram(raw_rotation_x,48,0,[],50,'MinThreshold',-20,'yaxis');
% view(-77,72)
% shading interp
% colorbar off

figure
spectrogram(new_rotation_x,48,0,[],50,'MinThreshold',-25,'yaxis');
% view(-77,72)
% shading interp
colorbar off

