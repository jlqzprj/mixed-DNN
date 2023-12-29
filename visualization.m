x = linspace(0,1,201); %for ex5.1, 5.2, and 5.7 change this line to linspace(-1,1,201);
y = x;
[X1,X2] = meshgrid(x,y);
X1 = X1';
X2 = X2';

load('YOUR_PATH/YOUR_INPUT_FILE_NAME','sigma_eval1')
load('YOUR_PATH/YOUR_OUTPUT_FILE_NAME','sigma')

figure(1), s1 = surf(X1,X2,reshape(sigma_eval1,size(X1)));
s1.LineStyle = 'none';
axis off
colormap(parula)
colorbar;

figure(2), s2 = surf(X1,X2,reshape(sigma,size(X1)));
s2.LineStyle = 'none';
axis off
colormap(parula)
colorbar;

figure(3), s3 = surf(X1,X2,reshape(abs(sigma-sigma_eval1),size(X1)));
s3.LineStyle = 'none';
axis off
colormap(jet)
colorbar;
