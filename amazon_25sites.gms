option limrow = 0;
option limcol = 0;
option solprint = off;

set T ordered /0*200 /;
set R ordered /1*25/;

scalar scale / 1.0e6 /;

parameter x0(R) ;
$call csv2gdx "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\X0Data.csv" output="C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\X0Data.gdx" id=dim1 index=1 values=1..lastCol useHeader=y trace=0
$gdxIn "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\X0Data.gdx"
$load  x0 = dim1

parameter z0(R) /
1   577.0929401
2   5589.185181
3   787674.0828
4   9646.861745
5   128634.5031
6   16248.65369
7   77127.05958
8   413615.3759
9   1415637.311
10  2975490.403
11  7911410.246
12  311951.5001
13  491859.8041
14  548889.5383
15  1193672.493
16  1416617.977
17  6055143.679
18  6746442.015
19  84410.49025
20  2448200.526
21  7385903.416
22  8456499.239
23  5225548.432
24  860690.5049
25  2311937.326
/;

parameter zbar(R) /
1   827842.673
2   5923957.508
3   16269303.58
4   8317106.102
5   11492722.45
6   3641993.119
7   27941871.29
8   26031526.11
9   26494123.96
10  25883433.49
11  18981310.08
12  986453.6577
13  23983732.12
14  28379164.51
15  27505450.39
16  28080696.69
17  27530403.94
18  9828455.437
19  4377913.526
20  8346530.632
21  21218696.15
22  20770966.56
23  16507489.38
24  1577160.408
25  3458325.658
/;


parameter gamma(R);
$call csv2gdx "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\GammaData.csv" output="C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\GammaData.gdx" id=dim1 index=1 values=1..lastCol useHeader=y trace=0
$gdxIn "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\GammaData.gdx"
$load  gamma = dim1


parameter theta(R) /
1   0.009738465
2   0.009740588
3   0.145421111
4   1.223269896
5   1.438574328
6   0.410137633
7   0.605161235
8   1.129249341
9   1.681938657
10  1.956043569
11  1.730931241
12  1.039779454
13  1.475119096
14  1.482043734
15  2.150337598
16  1.888581761
17  1.922081002
18  2.117680047
19  2.226926842
20  2.382020709
21  2.98371336
22  2.606553494
23  2.120503281
24  2.701969861
25  2.662808546
/;

parameter delta / 0.02 /;
parameter p_e / 20.76 /;
    parameter p_a / 44.75/;

parameter alpha / 0.045007414 /;
parameter kappa / 2.094215255 /;
parameter zeta / 1.66e-04 /;
parameter dt /1 /; 


parameter x_agg(T);
parameter z_agg(T);
parameter u_agg(T);
parameter v_agg(T);
parameter c_agg(T);

positive variable z(T,R);
positive variable u(T,R);
positive variable v(T,R);
         variable x(T,R);
         variable w(T);
         variable obj;

equation zdot_def(T,R);
equation xdot_def(T,R);
equation w_def(T);
equation obj_def;

zdot_def(T,R)$(ord(T) < card(T))..
 ( z(T+1,R) - z(T,R))/dt =e= (u(T,R) - v(T,R));

xdot_def(T,R)$(ord(T) < card(T))..
  (x(T+1,R) - x(T,R))/dt =e= (-gamma(R)*u(T,R) - alpha*x(T,R) + alpha*gamma(R)*(zbar(R)/scale - z(T,R)));

w_def(T)$(ord(T) < card(T))..
  w(T) =e= sum(R, u(T,R) + v(T,R));

obj_def..
  obj =e= sum(T$(ord(T) < card(T)), exp(-delta*(ord(T)*dt-dt))*(-p_e*sum(R, kappa*z(T,R) - (x(T+1,R) - x(T,R))/dt) + p_a*sum(R, theta(R)*z(T,R)) - scale*zeta/2*sqr(w(T)))*dt);
    
model amazon / all /;

file results_x / "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\amazon_data_x.dat" /;
file regionresults_z / "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\amazon_data_z.dat" /;

file regionresults_u / "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\amazon_data_u.dat" /;
file regionresults_v / "C:\Users\Samuel Zhao\Documents\GitHub\HMC_amazon_project2\amazon_data_v.dat" /;

regionresults_u.pw = 163840000;
regionresults_v.pw = 163840000;
regionresults_z.pw = 163840000;
x.fx(T,R)$(ord(T) = 1) = x0(R) / scale;	
z.fx(T,R)$(ord(T) = 1) = z0(R) / scale;
z.up(T,R)$(ord(T) > 1) = zbar(R) / scale ;
u.fx(T,R)$(ord(T) = card(T)) = 0;
v.fx(T,R)$(ord(T) = card(T)) = 0;
w.fx(T)$(ord(T) = card(T)) = 0;

option qcp = cplex;
solve amazon using qcp maximizing  obj;

x_agg(T) = scale*sum(R, x.l(T,R));
z_agg(T) = scale*sum(R, z.l(T,R));
u_agg(T) = scale*sum(R, u.l(T,R));
v_agg(T) = scale*sum(R, v.l(T,R));
c_agg(T) = scale*scale*sum(R, u.l(T,R)*v.l(T,R));

put results_x;
put 'T':4 system.tab 'x_agg':20/;
loop(T, put T.tl:4 system.tab x_agg(T):20:16 /);
putclose;

put regionresults_z;
put 'T/R':4;
loop(R, put system.tab R.tl:10);
put /;
loop(T,
  put T.tl:4;
  loop(R, put system.tab (scale*z.l(T,R)):16:10);

  put /;
);
putclose;

put regionresults_u;
put 'T/R':4;
loop(R, put system.tab R.tl:10);
put /;
loop(T,
  put T.tl:4;
  loop(R, put system.tab (scale*u.l(T,R)):16:10);

  put /;
);
putclose;

put regionresults_v;
put 'T/R':4;
loop(R, put system.tab R.tl:10);
put /;
loop(T,
  put T.tl:4;
  loop(R, put system.tab (scale*v.l(T,R)):16:10);

  put /;
);
putclose;

