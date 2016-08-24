import numpy as np

def hamdist(str1, str2):
    """Count the # of differences between equal length strings str1 and str2
    """
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

def calc_distance(quer, retr):
    """Calculate hamming distance
    
    args:
        quer: query set matrix [number,binary_length]
        retr:  retrieval set matrix [number,binary_length]
        
    return:
        dist: distance every pairs
    """
    x1,foox = quer.shape
    y1,fooy = quer.shape
    
    assert (foox==fooy),"vectors have different dimensions"
    
    dist = np.zeros([x1,y1])
    for i in xrange(x1):
        for j in xrange(y1):
            dist[i,j] = hamdist(quer[i],retr[j])
    return dist
    
def ir_perquery(gt_quer, distance_mtx, gt_retr):
    """ Calculate related information about retrieval
    
    args:
        distance_mtx: Distance of query from the rest of the datapoints
                                Each row refers to one query
        ground_truth: A ground truth vector of categories
    
    per_query is a dictionary with following elements:
        map: Mean average precision, for all the queries
        apr : dictionary with the following two fields:
            P: Average P/n for all the queries
            R: Average R/n for all the queries
                use the last two to compute P/R and P/n curves
        pr: precision recall, to plot against 0:0.001:1
        cm: confusion matrix
        
    per_class is a dictionary with following elements:
        map: Mean average precision, average over the classes
        map_class: vector of Mean Average Precision, for all class
        
        apr: dictionary with the following two field:
            P: Average P/n curve, averaged over the classes
            R: Average R/n curve, averaged over the classes
                use the last two together to build the PR curve
        apr_class: list of structs(as many as the classes) with the 
            following fields
            P: Average P/n curve, for one class
            P: Average R/n curve, for one class
            
    apr_class
        pr: matrix, each row contains the pr curve for a class,
            to be plotted vs 0:0.001:1
    """
    r,c = distance_mtx.shape
    testPoints = c
    queryPoints = r
    
    # Number of ground_truth categories
    cat_num = gt_quer.shape[1]
    assert (gt_quer.shape[1]==gt_retr.shape[1]),"Groud_truth of query's category not equal to retrieval's!"
    
    # Cardinality of the categories
    cat_card = np.zeros([1,cat_num])
    cat_card = np.sum(gt_retr,axis=0).astype(np.int32)
    
    ROCarea = np.zeros([1,testPoints])
    # Rank accuracy
    MAP = np.zeros([queryPoints,cat_num])
    
    top_5 = np.zeros([1,testPoints])
    top_10 = np.zeros([1,testPoints])
    top_20 = np.zeros([1,testPoints])
    top_40 = np.zeros([1,testPoints])
    top_60 = np.zeros([1,testPoints])
    top_100 = np.zeros([1,testPoints])
    
    # Confusion matrix
    conf = np.zeros([cat_num])
    
    # Precision and Recall
    pr = []
    P = np.zeros([queryPoints,testPoints])
    R = np.zeros([queryPoints,testPoints])
    pn = np.zeros([queryPoints,testPoints])
    
    # R-Precision
    rprecision = np.zeros([queryPoints])
    
    truemap = np.zeros([queryPoints])
    
    for itext in xrange(queryPoints):
        dist = distance_mtx[itext,:]
        foo = np.sort(dist)
        ind = np.argsort(dist)
        
        # most similar image (take the original index)
        # Pick the class to which the query belongs to
        for cls in xrange(cat_num):
            classes = cls
            
            # Classes of all queries, from best to worst match
            classesT = gt_retr[ind]
            
            # Make 0-1 GT
            classesGT = classesT[:,classes]
            
            # Compute the indexes in the rank
            ranks = np.where(classesGT>0)[0]
            ranks += 1
            
            # Compute AP for the query
            Map = np.sum(np.array([i+1 for i in xrange(len(ranks))]).astype(float)/ranks)/len(ranks)
            
            if np.isfinite(Map):
                MAP[itext,cls]=Map
        # Change1: truemap(itext) = MAP()        
        classeT = gt_retr[ind]
        rank = []
        classeGT = np.zeros([testPoints])
        for i in xrange(testPoints):
            for j in xrange(cat_num):
                if classeT[i,j]==gt_quer[itext,j] and gt_quer[itext,j]==1:
                    rank.append(i+1)
                    classeGT[i] = 1
                    break
        ranks = np.array(rank)
#        print "ranks"
#        print ranks
#        print np.array([i+1 for i in xrange(len(ranks))]).astype(float)
        truemap[itext] = np.sum(np.array([i+1 for i in xrange(len(ranks))]).astype(float)/ranks)/len(ranks)
        
        classe = gt_quer[itext]
        pn[itext,:] = np.cumsum(classeGT) /np.array([i+1 for i in xrange(testPoints)])
        
        idx = np.where(classe>0)[0]
#	print "index"
#	print idx
#	print "cat_card"
#	print cat_card[idx]
#	print "pn"
#	print pn[itext,cat_card[idx]]
#	print "itext"
#	print itext
#	print classe
        rp_max = max(pn[itext,cat_card[idx]])
        if rp_max>0:
            rprecision[itext] = rp_max
        else:
            rprecision[itext] = float("nan")
     
    cm = np.zeros([cat_num,cat_num])
            
    for cls in xrange(cat_num):
        if np.sum(gt_quer[:,cls])>0:
            cm[cls,:] = np.mean(MAP[np.where(gt_quer[:,cls]>0)[0],:])
            
    query = {}
    query["map"] = np.mean(truemap)
    query["cmmap"] = cm
    query["cm"] = cm
    query["pn"] = pn
    query["querymap"] = truemap
#    print truemap
    query["MAP"] = MAP
    # R-Precision
    query["queryrprecision"] = rprecision
    query["rprecision"] = np.mean(rprecision)
    
    cla = {}
    cla["map_class"] = np.diag(cm)
    return query,cla

def getprfrompn(pn,mode):
    # mode 0: interpolated
    n,c = pn.shape
    pnround = np.round(pn*np.tile(np.array([i+1 for i in xrange(c)]),(n,1)))
    al = []
    pr = []
    
    if mode == 0:
        for i in xrange(n):
            l = pn[i,:]
            p = pnround[i,:]
            curr = l[-1]
            for j in range(l.shape[1]-1,-1,-1):
                if l[j]<curr:
                    l[j] = curr
                else:
                    curr = l[j]
            up = []
            upid = []
            #for j in range(max(p)-1:-1:-1):
               # f = np.where(p==j)[0]
                #up.append(l[f[0]])
                #up.append(l[f[0]])
                #upid.append(j)
                #upid.append(j-0.99999)
                
                


def retrieval(quer, gt_q, retr, gt_r, rem):
    """retrieval 
    
    args:
        quer: query set (one sample per row)
        gt_q:  groundtruth for the query set
        
        retr:  retrieval set (one sample per row)
        gt_r:  groundtruth for the retrieval set
        
        rem: remove diagonal from distance mtx
                Note: remove is useful if: quer == retr
                
    return:
        
    """
    PREC_K = 10
    # Calculate all distance every pairs
    distAll = calc_distance(quer,retr)
    dist_lower_dim = min(distAll.shape)
    
    # Remove diagonal distance if rem==True
    if rem:
        for s  in xrange(dist_lower_dim):
            distAll[s,s] = float('Inf')
            
    # Compute IR metris retreval
    query,cla = ir_perquery(gt_q,distAll+1,gt_r)
    Q = query
    C = cla
    
    # mAP/P@k/R-Rrecision
    M_map = query["map"]
    M_prec_at_k = np.mean(query["pn"][:,PREC_K])
    M_rprecision = query["rprecision"]
    
    info = {}
    info["mAP"] = M_map
    info["mAP_per_class"] = C["map_class"]
    info["Precision_at_k"] = M_prec_at_k
    info["k"] = PREC_K
    info["RP"] = M_rprecision
    info["dist"] = distAll
    return Q,C,info
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
