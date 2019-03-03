My code are adapted from the github repos:
    https://github.com/cmaron/CS-7641-assignments/tree/master/assignment2
    https://github.com/mitian223/CS7641/tree/master/HW2


Requirement:
install jython 2.7
install ant
install ABAGAIL 
    it should work if you just download this repo,if not, 
    make sure you have java installed and re-compile the source code in the ABAGAIL folder with ant
        cd ./ABAGAIL
        ant
install numpy, pandas, matplotlib, sklearn, jupyter notebook


Instruction:
if you want to resplit the data(generate new training , testing sets)
    run python bank_spliter.py

if you want to run the neural network problem
    run jython ANN.py
    results are in the ANN folder

if you want to generate figures for neural network problem
    use jypyter notebook to run ANN_PLOT.ipynb
    images are in the ANN folder

if you want to run the flip flop, continuous peakk, traveling salesman problem
    run jython flipflop.py
        jython coutinuouspeaks.py
        jython tsp.py
    results are in the output folder

if you want to generate the plot for these problems
    run python plotting.py
    images are in the output/images folder

And I also upload the results for my experiment. ANN_1000 strores the results when i use fewer iterations.