from retrieval import retrieval
import numpy as np
a = np.array([[-1,-1,-1,-1,-1,-1,-1,-1],
                     [-1,-1,-1,-1,-1,-1,-1,1],
                     [-1,-1,-1,-1,-1,-1,1,-1],
                     [-1,-1,-1,-1,-1,1,-1,-1],
                     [1,1,1,1,1,1,1,1],
                     [1,1,1,1,1,1,1,-1],
                     [1,1,1,1,1,1,-1,1],
                     [1,1,1,1,1,-1,1,1]])
b = np.array([[1,0],
                     [1,0],
                     [1,0],
                     [1,0],
                     [0,1],
                     [0,1],
                     [0,1],
                     [0,1]]) 
Q,C,info = retrieval(a,b,a,b,True)
print "map"
print Q["map"]
print "cm"
print Q["cm"]
print "querymap"
print Q["querymap"]
print "MAP"
print Q["MAP"]
print "cmmap"
print Q["cmmap"]
print "rprecision"
print Q["rprecision"]
print "pn"
print Q["pn"]
print "queryrprecision"
print Q["queryrprecision"]
print "--------------------------------------------"
print C["map_class"]
print "**********************"
print "mAP"
print info["mAP"]
print "distance"
print info["dist"]
print "mAP_per_class"
print info["mAP_per_class"]
print "RP"
print info["RP"]
print "Precision_at_k"
print info["Precision_at_k"]