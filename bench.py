from pymorph import *

def bench(count=10):
    """
        - Purpose
            benchmarking main functions of the toolbox.
        - Synopsis
            bench(count=10)
        - Input
            count: Double Default: 10. Number of repetitions of each
                   function.

        - Description
            bench measures the speed of many of SDC Morphology Toolbox
            functions in seconds. An illustrative example of the output of
            bench is, for a MS-Windows 2000 Pentium 4, 2.4GHz, 533MHz
            system bus, machine: SDC Morphology Toolbox V1.2 27Sep02
            Benchmark Made on Wed Jul 16 15:33:17 2003 computer= win32 image
            filename= csample.jpg width= 640 , height= 480 Function time
            (sec.) 1. Union bin 0.00939999818802 2. Union gray-scale
            0.00319999456406 3. Dilation bin, secross 0.0110000014305 4.
            Dilation gray, secross 0.00780000686646 5. Dilation gray,
            non-flat 3x3 SE 0.0125 6. Open bin, secross 0.0125 7. Open
            gray-scale, secross 0.0141000032425 8. Open gray, non-flat 3x3
            SE 0.0235000014305 9. Distance secross 0.021899998188 10.
            Distance Euclidean 0.0264999985695 11. Geodesic distance
            secross 0.028100001812 12. Geodesic distance Euclidean
            0.303100001812 13. Area open bin 0.0639999985695 14. Area open
            gray-scale 0.148500001431 15. Label secross 0.071899998188 16.
            Regional maximum, secross 0.043700003624 17. Open by rec,
            gray, secross 0.0515000104904 18. ASF by rec, oc, secross, 1
            0.090600001812 19. Gradient, gray-scale, secross
            0.0171999931335 20. Thinning 0.0984999895096 21. Watershed
            0.268799996376 Average 0.0632523809161

    """
    from sys import platform
    from time import time, asctime
    from numpy import average, zeros

    filename = 'csample.jpg'
    f = readgray(filename)
    fbin=threshad(f,150)
    se = img2se(binary([[0,1,0],[1,1,1],[0,1,0]]),'NON-FLAT',to_int32([[0,1,0],[1,2,1],[0,1,0]]))
    m=thin(fbin)
    tasks=[
       [' 1. Union  bin                      ','union(fbin,fbin)'],
       [' 2. Union  gray-scale               ','union(f,f)'],
       [' 3. Dilation  bin, secross        ','dilate(fbin)'],
       [' 4. Dilation  gray, secross       ','dilate(f)'],
       [' 5. Dilation  gray, non-flat 3x3 SE ','dilate(f,se)'],
       [' 6. Open      bin, secross        ','open(fbin)'],
       [' 7. Open      gray-scale, secross ','open(f)'],
       [' 8. Open      gray, non-flat 3x3 SE ','open(f,se)'],
       [' 9. Distance  secross             ','dist(fbin)'],
       ['10. Distance  Euclidean             ','dist(fbin,sebox(),"euclidean")'],
       ['11. Geodesic distance secross     ','gdist(fbin,m)'],
       ['12. Geodesic distance Euclidean     ','gdist(fbin,m,sebox(),"euclidean")'],
       ['13. Area open bin                   ','areaopen(fbin,100)'],
       ['14. Area open gray-scale            ','areaopen(f,100)'],
       ['15. Label secross                 ','label(fbin)'],
       ['16. Regional maximum, secross     ','regmax(f)'],
       ['17. Open by rec, gray, secross    ','openrec(f)'],
       ['18. ASF by rec, oc, secross, 1    ','asfrec(f)'],
       ['19. Gradient, gray-scale, secross ','gradm(f)'],
       ['20. Thinning                        ','thin(fbin)'],
       ['21. Watershed                       ','cwatershed(f,fbin)']]
    result = zeros((21),'d')
    for t in xrange(len(tasks)):
       print tasks[t][0],tasks[t][1]
       t1=time()
       for k in xrange(count):
          a=eval(tasks[t][1])
       t2=time()
       result[t]= (t2-t1)/(count+0.0)
    print version() +' Benchmark'
    print 'Made on ',asctime(),' computer=',platform
    print 'image filename=',filename,' width=', f.shape[1],', height=',f.shape[0]
    print '    Function                            time (sec.)'
    for j in xrange(21):
     print tasks[j][0], result[j]
    print '    Average                         ', average(result)
    out=[]
