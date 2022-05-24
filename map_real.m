clc
clear all

x = importdata('vector_all_samson_pre.mat');


figure(1)
for i = 1:3
    subplot(1,3,i)
    imagesc(reshape(x(:,i),[95,95])',[0,1])
end
 
y = importdata('endmember_pre_samson.mat');
figure(2)
subplot(1,3,1)
plot(y(1,1:156))
subplot(1,3,2)
plot(y(2,157:157+156))
subplot(1,3,3)
plot(y(3,156*2+1:156*3))