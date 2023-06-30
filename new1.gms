$title CSV2GDX Example 2 - Reading CSV Files with CSV2GDX (CSV2GDX2,SEQ=111)

$onText
This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories.

CSV2GDX is used to read all data from external CSV files
with comma separated fields.
Note: In this example, the CSV data can also be read by using a simple
table statement without doing further preprocessing.

This model is referenced in "Getting Started Example 1 - Reading
CSV files with CSV2GDX" from the CSV2GDX Documentation.

Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

This formulation is described in detail in:
Rosenthal, R E, Chapter 2: A GAMS Tutorial. In GAMS: A User's Guide.
The Scientific Press, Redwood City, California, 1988.

The line numbers will not match those in the book because of these
comments.

Keywords: linear programming, transportation problem, scheduling,
          CSV2GDX, data exchange, GAMS language features
$offText

$onEcho > distance.csv
,new-york,chicago,topeka
seattle,2.5,1.7,1.8
san-diego,2.5,1.8,1.4
$offEcho

$onEcho > capacity.csv
,capacity
seattle,350
san-diego,600
$offEcho

$onEcho > demand.csv
,demand
new-york,325
chicago,300
topeka,275
$offEcho

Set
   i 'canning plants'
   j 'markets';

$call csv2gdx distance.csv id=d index=1 values=2..lastCol useHeader=y trace=0
$ifE errorLevel<>0 $abort Problems reading distance.csv!
$gdxIn distance.gdx
$load i = dim1
$load j = dim2

Parameter d(i,j) 'distance in thousands of miles';
$load d
$gdxIn
display i, j, d;

Parameter
   a(i) 'capacity of plant i in cases'
   b(j) 'demand at market j in cases';

* Since we are not interested of the information in the header in this case,
* useHeader=y can be replaced by colCount=2 in both calls.
$call csv2gdx capacity.csv id=a index=1 useHeader=y value=2 trace=0
$ifE errorLevel<>0 $abort Problems reading capacity.csv!
$gdxIn capacity.gdx
$load a
$gdxIn

$call csv2gdx demand.csv id=b index=1 useHeader=y value=2 trace=0 
$ifE errorLevel<>0 $abort Problems reading demand.csv!
$gdxIn demand.gdx
$load b
$gdxIn
display a, b;

Scalar f 'freight in dollars per case per thousand miles' / 90 /;

Parameter c(i,j) 'transport cost in thousands of dollars per case';
c(i,j) = f*d(i,j)/1000;

Variable
   x(i,j) 'shipment quantities in cases'
   z      'total transportation costs in thousands of dollars';

Positive Variable x;

Equation
   cost      'define objective function'
   supply(i) 'observe supply limit at plant i'
   demand(j) 'satisfy demand at market j';

cost..      z =e= sum((i,j), c(i,j)*x(i,j));

supply(i).. sum(j, x(i,j)) =l= a(i);

demand(j).. sum(i, x(i,j)) =g= b(j);

Model transport / all /;

solve transport using lp minimizing z;

display x.l, x.m;