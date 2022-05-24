clc
clear all
M = importdata('end_ext_samson.mat');
weight_samson_long =zeros(156*3,3);
weight_samson_long(1:156,1) = M(:,1);
a=156*2;
weight_samson_long(157:a,2) = M(:,2);
b=a+156;
weight_samson_long(a+1:b,3) = M(:,3);
weight_samson_long=single(weight_samson_long);
save weight_samson_long weight_samson_long