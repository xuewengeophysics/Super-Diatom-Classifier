w=100;
h=100;
sigma2=30^2;
Acc=zeros(h,w);
AccW=zeros(h,w);
xMap=ones(h,1)*[1:w];
yMap=[1:h]'*ones(1,w);
Data=[20,20,255;
    20,80,128;
    80,80,92];
for i=1:3
    d2 = (xMap - Data(i,1)).^2 + (yMap - Data(i,2)).^2;
    w = max(exp(-0.5*d2/sigma2),1e-10);
    Acc = Acc + Data(i,3)*w;
    AccW = AccW + w;
end

Acc = Acc ./ AccW;
imagesc(Acc)