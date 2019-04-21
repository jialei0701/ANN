% 给定图片，绘制曲面图。用于输入图片仿真。

function f(p,iter,loss)
  t=imresize(p, 0.25);
  r=flipud(t(:,:,1));
  xa=1:1:90;
  ya=1:1:68;
  [x,y]=meshgrid(xa,ya);
  figure;
  subplot('position', [.1 .3 .8 .6])
  surf(x,y,r)
  view(-14,72)
  set(gcf, 'position', [0 0 900 660]);
  title("iter="+iter+", loss="+loss)
  subplot('position', [.1 .03 .8 .2])
  imshow(p)
return
