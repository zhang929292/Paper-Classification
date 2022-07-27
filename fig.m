clc
clear all

data = importdata('cora/cora.cites');
[m, n] = size(data);
for i = 1:m
    for j = 1:n
        str{i,j} = num2str(data(i,j));
    end
end

G2 = graph(str(:,1)', str(:,2)');
plot(G2, 'linewidth', 2)  

