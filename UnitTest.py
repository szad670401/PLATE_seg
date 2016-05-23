__author__ = 'yujinke'
import numpy as np;

def findSmallPeak(points,fixed):
    points = np.array(points)
  #  min  =  points.argmin()
    c_pos = fixed;
    step = 1;
    epoch = 20 ;

    for i in range(epoch):

        prev = points[c_pos - step];
        back = points[c_pos + step];
        if((c_pos - step) >= 0 and c_pos + step < len(points)):


            if(prev > back):
                symbol = 1;
            if(prev <= back):
                symbol = -1;
            c_pos+= symbol;
            print c_pos


    return c_pos;


print findSmallPeak([0.3,0.26,0.23,0.22,0.20,0.11,0.2,0.11,0.22,0.21,0.22,0.3],6);

