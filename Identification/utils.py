''' Import '''
from collections import namedtuple
from urllib.parse import urlparse
import logging
import torch
import numpy as np
import pandas as pd
import os,sys
import time
import math
import copy
from pbcore.io.opener import (openAlignmentFile, openIndexedAlignmentFile)
import warnings
from itertools import chain
import unidip.dip as dip
warnings.filterwarnings("ignore")
import scipy.stats as stats
import re 
from recordtype import recordtype
from scipy.stats import ks_2samp
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import datetime

''' predefine. '''
ipdRec = [('tpl','<u4'),
        ('zmwhole','<u4'),
        ('strand','<i8'),
        ('rv_ipd','<f4')]
feature = recordtype('feature',['ID',
'kmerList','ipdList'])
ReferenceWindow = namedtuple(
    "ReferenceWindow", ["refId", "refName", "start", "end"])
resRec = [('tpl','<u4'),
        ('strand','<i8'),
          ('sigQ','<u4'),
        ('coverage','<u4'),
         ('shufQ','<u4'),
         ('shufCov','<u4')]

''' Func. '''
#region
def _validateResource(func, p):
    """Basic func for validating files, dirs, etc..."""
    if func(p):
        return os.path.abspath(p)
    else:
        raise IOError("Unable to find {p}".format(p=p))

def _validateNoneOrResource(func, p):
    """
    Handle optional values. If a file or dir is explicitly provided, then
    it will validated.
    """
    if p is None:
        return p
    else:
        return _validateResource(func, p)

def _openFiles(self, refFile=None, sharedIndices=None):
    """
    Hack to enable sharing of indices (but not filehandles!) between dataset
    instances.
    """
    log = logging.getLogger()
    log.debug("Opening resources")
    for k, extRes in enumerate(self.externalResources):
        location = urlparse(extRes.resourceId).path
        sharedIndex = None
        if sharedIndices is not None:
            sharedIndex = sharedIndices[k]
        try:
            resource = openIndexedAlignmentFile(
                location,
                referenceFastaFname=refFile,
                sharedIndex=sharedIndex)
        except (IOError, ValueError):
            log.info("pbi file missing for {f}, operating with "
                     "reduced speed and functionality".format(
                         f=location))
            resource = openAlignmentFile(location,
                                         referenceFastaFname=refFile)
        if len(resource) == 0:
            log.warn("{f} has no mapped reads".format(f=location))
        else:
            self._openReaders.append(resource)
    if len(self._openReaders) == 0:
        raise IOError("No mapped reads found")
    log.debug("Done opening resources")

def _reopen(file):
    """
    Force re-opening of underlying alignment files, preserving the
    reference and indices if present, and return a copy of the
    AlignmentSet.  This is a workaround to allow us to share the index
    file(s) already loaded in memory while avoiding multiprocessing
    problems related to .bam files.
    """
    refFile = file._referenceFile
    newSet = copy.deepcopy(file)
    newSet._referenceFastaFname = refFile
    indices = [f.index for f in file.resourceReaders()]
    file.close()
    _openFiles(newSet, refFile=refFile, sharedIndices=indices)
    return newSet

def markRep(array):
        uniq_bool = np.zeros(len(array),dtype = bool)
        uniq_bool[len(array)-1] = uniq_bool[0] = True
        for x in range(1,len(array)-1):
            if array[x] == array[x+1] or array[x] == array[x-1]:
                uniq_bool[x] = uniq_bool[x+1] = False
            else:
                uniq_bool[x] = True
                pass
        return uniq_bool

def test(group1,group2):
    try:
        u_statistic, pVal = stats.mannwhitneyu(group1, group2, \
                                               nan_policy='omit',\
                                               alternative='greater')
        return round(pVal,4)
    except:
        return np.nan

def fold_change_median(x,y):
    if np.nanmedian(y) == 0:
        fold_change = np.nan
    else:
        fold_change=math.log((np.nanmedian(x)/np.nanmedian(y)+1e-100),2)
    return round(fold_change,4)
  
def postFilter(hits_dict,cutoff = 5):
    for i in list(hits_dict):
        if len(hits_dict[i]) < cutoff:
            del hits_dict[i]
    return hits_dict
                 
def parseReferenceWindow(s, refInfoLookup):
    if s is None:
        return None
    m = re.match("(.*):(.*)-(.*)", s)
    if m:
        refContigInfo = refInfoLookup(m.group(1))
        refId = refContigInfo.ID
        refName = refContigInfo.Name
        refStart = int(m.group(2))
        refEnd = min(int(m.group(3)), refContigInfo.Length)
    else:
        refContigInfo = refInfoLookup(s)
        refId = refContigInfo.ID
        refName = refContigInfo.Name
        refStart = 0
        refEnd = refContigInfo.Length
    return ReferenceWindow(refId=refId, refName=refName, start=refStart,
                           end=refEnd)

def createReferenceWindows(refInfo):
    return [ReferenceWindow(refId=r.ID,
                            refName=r.Name,
                            start=0,
                            end=r.Length) for r in refInfo]

def enumerateChunks(referenceStride, referenceWindow):
    """
    Enumerate all work chunks on this reference contig (restricted to
    the windows, if provided).
    """
    def intersection(int1, int2):
        s1, e1 = int1
        s2, e2 = int2
        si, ei = max(s1, s2), min(e1, e2)
        if si < ei:
            return (si, ei)
        else:
            return None

    def enumerateIntervals(bounds, stride):
        """
        Enumerate windows of size "stride", attempting to align window
        boundaries on multiple of stride.
        """
        def alignDown(chunk, x):
            return (x // chunk) * chunk

        def alignUp(chunk, x):
            return int(math.ceil(float(x) / chunk) * chunk)

        start, end = bounds
        roundStart = alignDown(stride, start)
        roundEnd = alignUp(stride, end)

        for s in range(roundStart, roundEnd, stride):
            roundWin = (s, s + stride)
            yield intersection(bounds, roundWin)

    for (s, e) in enumerateIntervals((referenceWindow.start,
                                      referenceWindow.end), referenceStride):
        yield ReferenceWindow(refId=referenceWindow.refId,
                              refName=referenceWindow.refName,
                              start=s, end=e)

def referenceWindowsFromAlignment(ds, refInfoLookup):
    return [ReferenceWindow(refId=refInfoLookup(w[0]).ID,
                            refName=w[0],
                            start=w[1],
                            end=w[2]) for w in ds.refWindows]

def monitorChildProcesses(children):
    """
    Monitors child processes: promptly exits if a child is found to
    have exited with a nonzero exit code received; otherwise returns
    zero when all processes exit cleanly (0).

    This approach is portable--catching SIGCHLD doesn't work on
    Windows.
    """
    while True:
        all_exited = all(not p.is_alive() for p in children)
        nonzero_exits = [p.exitcode for p in children if p.exitcode]
        if nonzero_exits:
            exitcode = nonzero_exits[0]
            logging.error(
                "Child process exited with exitcode=%d.  Aborting." % exitcode)

            # Kill all the child processes
            for p in children:
                if p.is_alive():
                    p.terminate()

            os._exit(exitcode)
        elif all_exited:
            return 0
        time.sleep(1)

def coden_complement(code_list):
    complement_dict = {'A':'T','T':'A','G':'C','C':'G',\
                'a':'t','t':'a','g':'c','c':'g','-':'-'}
    complement_list = np.vectorize(complement_dict.get)(code_list)
    return np.array(complement_list)

def calculate_RV(A_vector,N_vector):
    func = lambda x,y: round(2*x/(x+y),4) if y!=np.nan and (x+y)!=0 else np.nan
    RV_tmp = [func(x,y) for (x,y) in zip(A_vector,N_vector)]
    return np.array(RV_tmp)

def caseMerged(case_hits_pool):
    rv_hits_merged = {}
    for site in case_hits_pool:
        newIdx = (site['tpl'],site['strand'],site['zmwhole'])
        if newIdx not in rv_hits_merged.keys():
            rv_hits_merged[newIdx] = [site['rv_ipd']]
        else:
            rv_hits_merged[newIdx].append(site['rv_ipd'])
    return rv_hits_merged

def controlMerged(control_hits_pool):
    rv_chits_merged = {}
    for site in control_hits_pool:
        newIdx = (site['tpl'],site['strand'])
        if newIdx not in rv_chits_merged.keys() :
            rv_chits_merged[newIdx] = [site['rv_ipd']]
        else:
            rv_chits_merged[newIdx].append(site['rv_ipd'])
    return rv_chits_merged

def RV_for_hit(hit,targetBound,dis = 10):
    try:
        if hit.isMapped == True:
            thit = Hit_trimming(hit,targetBound)
            uniq_bool = markRep(thit.referencePositions())
            read = np.array(coden_complement(list(thit.read())))[uniq_bool]
            tpl = thit.referencePositions()[uniq_bool]
            ipd = thit.IPD()[uniq_bool]
            A_index = np.where(np.logical_or((read == "A"),(read == "a")))[0]
            T_index = np.where(np.logical_or((read == "T"),(read == "t")))[0]
            G_index = np.where(np.logical_or((read == "G"),(read == "g")))[0]
            TG_index = np.concatenate((T_index,G_index))
            nearest_TG_index = []
            set_TG = set(TG_index)
            for i in A_index:
                Bound = range(i-dis,i+dis)
                intersection = list(set(Bound)&set_TG)
                if len(intersection) != 0:
                    nearest_TG = min(intersection,key = lambda x : abs(x - i))
                    nearest_TG_index.append(nearest_TG)
                else:
                    nearest_TG_index.append(np.nan)
            A_vector = ipd[A_index]
            Return = lambda i,array : array[i] if i is not np.nan else np.nan
            N_vector = np.array([Return(x,ipd) for x in nearest_TG_index])
            RV_array = calculate_RV(A_vector,N_vector)
            array = np.zeros(len(A_index), dtype=ipdRec)
            array['tpl'] = tpl[A_index]
            array['strand'] = hit.isReverseStrand
            array['zmwhole'] = hit.HoleNumber
            array['rv_ipd'] = RV_array
            return array 
        else:
            pass
    except:
            pass
   
def doTest(rv_hits_merged,rv_chits_merged):
    
    hits_merged_key_al = {}
                
    for x in rv_hits_merged.keys():
        if (x[0],x[1]) not in hits_merged_key_al.keys():
            hits_merged_key_al[(x[0],x[1])] = [x]
        else:
            hits_merged_key_al[(x[0],x[1])].append(x)
                
    key_intersection = list(set(hits_merged_key_al.keys() & rv_chits_merged.keys()))      
          
    result = {}

    for i in key_intersection:
        for j in hits_merged_key_al[i]:
            try:
                case = rv_hits_merged[j]
                control = rv_chits_merged[i]
                result[j] = [test(case,control),\
                            fold_change_median(case,control)]
            except:
                pass  
    
    return result

def _openFiles(self, refFile=None, sharedIndices=None):
    """
    Hack to enable sharing of indices (but not filehandles!) between dataset
    instances.
    """
    log = logging.getLogger()
    log.debug("Opening resources")
    for k, extRes in enumerate(self.externalResources):
        location = urlparse(extRes.resourceId).path
        sharedIndex = None
        if sharedIndices is not None:
            sharedIndex = sharedIndices[k]
        try:
            resource = openIndexedAlignmentFile(
                location,
                referenceFastaFname=refFile,
                sharedIndex=sharedIndex)
        except (IOError, ValueError):
            log.info("pbi file missing for {f}, operating with "
                     "reduced speed and functionality".format(
                         f=location))
            resource = openAlignmentFile(location,
                                         referenceFastaFname=refFile)
        if len(resource) == 0:
            log.warn("{f} has no mapped reads".format(f=location))
        else:
            self._openReaders.append(resource)
    if len(self._openReaders) == 0:
        raise IOError("No mapped reads found")
    log.debug("Done opening resources")
                
def parseReferenceWindow(s, refInfoLookup):
    if s is None:
        return None
    m = re.match("(.*):(.*)-(.*)", s)
    if m:
        refContigInfo = refInfoLookup(m.group(1))
        refId = refContigInfo.ID
        refName = refContigInfo.Name
        refStart = int(m.group(2))
        refEnd = min(int(m.group(3)), refContigInfo.Length)
    else:
        refContigInfo = refInfoLookup(s)
        refId = refContigInfo.ID
        refName = refContigInfo.Name
        refStart = 0
        refEnd = refContigInfo.Length
    return ReferenceWindow(refId=refId, refName=refName, start=refStart,
                           end=refEnd)

def createReferenceWindows(refInfo):
    return [ReferenceWindow(refId=r.ID,
                            refName=r.Name,
                            start=0,
                            end=r.Length) for r in refInfo]

def enumerateChunks(referenceStride, referenceWindow):
    """
    Enumerate all work chunks on this reference contig (restricted to
    the windows, if provided).
    """
    def intersection(int1, int2):
        s1, e1 = int1
        s2, e2 = int2
        si, ei = max(s1, s2), min(e1, e2)
        if si < ei:
            return (si, ei)
        else:
            return None

    def enumerateIntervals(bounds, stride):
        """
        Enumerate windows of size "stride", attempting to align window
        boundaries on multiple of stride.
        """
        def alignDown(chunk, x):
            return (x // chunk) * chunk

        def alignUp(chunk, x):
            return int(math.ceil(float(x) / chunk) * chunk)

        start, end = bounds
        roundStart = alignDown(stride, start)
        roundEnd = alignUp(stride, end)

        for s in range(roundStart, roundEnd, stride):
            roundWin = (s, s + stride)
            yield intersection(bounds, roundWin)

    for (s, e) in enumerateIntervals((referenceWindow.start,
                                      referenceWindow.end), referenceStride):
        yield ReferenceWindow(refId=referenceWindow.refId,
                              refName=referenceWindow.refName,
                              start=s, end=e)

def referenceWindowsFromAlignment(ds, refInfoLookup):
    return [ReferenceWindow(refId=refInfoLookup(w[0]).ID,
                            refName=w[0],
                            start=w[1],
                            end=w[2]) for w in ds.refWindows]

def monitorChildProcesses(children):
    """
    Monitors child processes: promptly exits if a child is found to
    have exited with a nonzero exit code received; otherwise returns
    zero when all processes exit cleanly (0).

    This approach is portable--catching SIGCHLD doesn't work on
    Windows.
    """
    while True:
        all_exited = all(not p.is_alive() for p in children)
        nonzero_exits = [p.exitcode for p in children if p.exitcode]
        if nonzero_exits:
            exitcode = nonzero_exits[0]
            logging.error(
                "Child process exited with exitcode=%d.  Aborting." % exitcode)

            # Kill all the child processes
            for p in children:
                if p.is_alive():
                    p.terminate()

            os._exit(exitcode)
        elif all_exited:
            return 0
        time.sleep(1)

def Hit_trimming(hit,targetBound):
    start,end = targetBound
    thit = hit.clippedTo(start,end) # open on left, close on right.
    return thit

def siteInHit(sitesArray,hit):
    intersect = np.in1d(sitesArray['tpl'],\
                        hit.referencePositions(),\
                        assume_unique = True)
    return sitesArray[intersect]

def filterHits(aln,start,end,sector=10):
    matched = np.array([x != '-' for x in aln.read()])
    np.logical_and(np.logical_not(np.isnan(aln.IPD())),matched,out = matched)
    np.logical_and(np.logical_not(np.isnan(aln.PulseWidth())),matched,out = matched)
    referencePositions = aln.referencePositions()
    np.logical_and(referencePositions <= end, matched, matched)
    np.logical_and(referencePositions >= start, matched, matched)
    nm = matched.sum()
    if nm != 2*sector+1:
        return None
    else:
        ipd = aln.IPD()[matched]
        read = coden_complement(list(aln.read()))[matched]
        return ipd,read

def mode1(x):
    if len(x) != 0:
        values,counts = np.unique(x,return_counts = True)
        if (counts==max(counts)).sum() > 1:
            return "-"
        else:
            return values[counts.argmax()]
    else:
        return "-"
  
def update(ipd_list,read_list):
    read_list = np.array(read_list).T
    read_list_update = np.array([mode1(col) for col in read_list]) 
    return np.array(ipd_list),read_list_update

def correct_pvalues_for_multiple_testing(pvalues, correction_type = "Bonferroni"):                
    """                                                                                                   
    consistent with R - print correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05, 0.069, 0.07, 0.071, 0.09, 0.1]) 
    """
    from numpy import array, empty                                                                        
    pvalues = array(pvalues) 
    n = int(pvalues.shape[0])                                                                           
    new_pvalues = empty(n)
    if correction_type == "Bonferroni":                                                                   
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":                                                            
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        for rank, vals in enumerate(values):                                                              
            pvalue, i = vals
            new_pvalues[i] = (n-rank) * pvalue                                                            
    elif correction_type == "Benjamini-Hochberg":                                                         
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        values.reverse()                                                                                  
        new_values = []
        for i, vals in enumerate(values):                                                                 
            rank = n - i
            pvalue, index = vals                                                                          
            new_values.append((n/rank) * pvalue)                                                          
        for i in range(0, int(n)-1):  
            if new_values[i] < new_values[i+1]:                                                           
                new_values[i+1] = new_values[i]                                                           
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]                                                                                                                  
    return new_pvalues  

def resMerged(res_dict):
    res_dict_p_merged = {}
    res_dict_fc_merged = {}
    for site in res_dict.keys():
        newIdx = (site[0],site[1])
        if newIdx not in res_dict_p_merged.keys():
            res_dict_p_merged[newIdx] = [res_dict[site][0]]
            res_dict_fc_merged[newIdx] = [res_dict[site][1]]
        else:
            res_dict_p_merged[newIdx].append(res_dict[site][0])
            res_dict_fc_merged[newIdx].append(res_dict[site][1])
    return res_dict_p_merged,res_dict_fc_merged

def shuffledFCandQ(pvalues,fc,cutoff = 10):
    pvalues = np.array(pvalues)
    fc = np.array(fc)
    assert len(pvalues) == len(fc)
    shuf_idx = list(range(len(pvalues)))
    np.random.shuffle(shuf_idx)
    qvalues = [i*cutoff for i in pvalues[shuf_idx[:cutoff]]]
    return qvalues,fc[shuf_idx[:cutoff]]

def shuffleFCandQ(p_dict,fc_dict,cutoff = 10):
    res_dict_shufQ = {}
    res_dict_shufFC = {}
    for i in p_dict.keys():
        qvalues,fcs = shuffledFCandQ(p_dict[i],fc_dict[i],cutoff = cutoff)
        res_dict_shufQ[i] = qvalues
        res_dict_shufFC[i] = fcs
    return res_dict_shufQ,res_dict_shufFC

def returnStep1Res(res_dict_q,res_dict_fc,res_dict_shufQ,
                 res_dict_shufFC,cov_cutoff,sig_cutoff):
    ResArray = np.zeros(len(res_dict_q), dtype=resRec)
    tplList = []
    strandList = []
    coverageList = []
    sigQList = []
    shufQList = []
    for i in res_dict_q.keys():
        coverage = len(res_dict_q[i])
        valid = [res_dict_q[i][idx] for idx in range(len(res_dict_q[i])) if res_dict_fc[i][idx] > 0 
                and res_dict_fc[i][idx] < 1 and res_dict_q[i][idx] <= 0.05]
        shufvalid = [res_dict_shufQ[i][idx] for idx in range(len(res_dict_shufQ[i])) if res_dict_shufFC[i][idx] > 0 
                and res_dict_shufFC[i][idx] < 1 and res_dict_shufQ[i][idx] <= 0.05]
        tplList.append(i[0])
        strandList.append(i[1])
        coverageList.append(coverage)
        sigQList.append(len(valid))
        shufQList.append(len(shufvalid))
    ResArray['tpl'] = tplList
    ResArray['strand'] = strandList
    ResArray['coverage'] = coverageList
    ResArray['sigQ'] = sigQList
    ResArray['shufQ'] = shufQList
    ResArray['shufCov'] = cov_cutoff
    candiList = []
    for i in ResArray:
        if i[2] >= sig_cutoff:
            candiList.append(i)
    candiArray = np.array(candiList, dtype=resRec)
    return pd.DataFrame(ResArray),candiArray

def getInfoList(hits,site,sector = 10):

    siteStart = site[0] - sector
    siteEnd = site[0] + sector
    ipd_list = []
    read_list = []

    for h in hits:
        try:
            if h.isReverseStrand == site[1]:
                ipd,read = filterHits(h,siteStart,siteEnd,sector) 
                ipd_list.append(ipd.tolist())
                read_list.append(read.tolist())
        except:
            continue

    return ipd_list,read_list 

def Run(ipd_list,read_list,ID,num = 5):

    ipd_list_update,read_list_update = update(ipd_list,read_list)

    if len(ipd_list_update) >= num:
        idx = [x for x in range(len(ipd_list_update))]
        np.random.shuffle(idx)
        ipd_list_update = ipd_list_update[idx[:num],:]
        return read_list_update,ipd_list_update
    else:
        return None
    
def Extract(refName,targetSites,hits_pool,sector,num):
    
    featureList = []

    for hole in hits_pool.keys():

        hits = hits_pool[hole]

        intersectSites = siteInHit(targetSites,hits[0])

        if len(intersectSites) == 0:
            pass
        else:
            for site in intersectSites:
                ID = refName + "_" + str(site[1]) + "_" + str(site[0])
                ipd_list,read_list = getInfoList(hits,site,sector)
                result = Run(ipd_list,read_list,ID,num)
                if result is None:
                    pass
                else:
                    kmerList,ipdList = result
                    featureList.append(feature(ID = ID,
                    kmerList = kmerList,
                    ipdList = ipdList))
    return featureList

def random_fn(x):
    np.random.seed(datetime.datetime.now().second)

def inference_collate(batch):
    return {key:batch for key,batch in zip(['X','kmer'],default_collate(batch))}

class torchDataset_inference(Dataset):

    def __init__(self,X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = torch.Tensor(self.X[idx][0])
        return feature
 
class modelInference():
    def __init__(self,featureList,state_dict,modelPath,batch_size=1024):
        self.featureList = featureList
        self.state_dict = state_dict
        self.modelPath = modelPath
        self.batch_size = batch_size
        sys.path.append(modelPath)
        from model_ccs_attn import model
        self.model = model()
        self.model.load_state_dict(torch.load(state_dict))
        
        self.loc = {'A':0,
          'G':1,
          'T':2,
          'C':3}
       
    def encoder(self,seq,ipdList):
        encd = np.zeros((4,len(ipdList)))
        new_seq = []
        for idx,i in enumerate(seq):
            loca = self.loc[i]
            ipd = ipdList[idx]
            encd[loca,idx] = ipd
        return np.array(encd.tolist())
    
    def minUmax(self,array):
        norm = lambda x: (x - min(x))/(max(x) - min(x))
        return np.apply_along_axis(norm,1,array)

    def _run(self,i):
        ipdList = []
        if '-' in  i.kmerList or \
        np.isnan(i.ipdList).any() == True: 
            return (None,None)
        else:
            ipd_list = self.minUmax(i.ipdList)
            for j in ipd_list:
                ipdList.append(self.encoder(i.kmerList,j))
            ipdList = np.array(ipdList)
            return (ipdList,i.ID)
    
    def inference(self,dl,device = 'cpu'):
        """
        Run inference on unlabelled dataset
        """
        self.model.eval()
        all_y_pred = []
        with torch.no_grad():
            kmers = []
            for batch in dl:
                y_pred = self.model(batch)
                y_pred = y_pred.detach().cpu().numpy()
                all_y_pred.extend(y_pred.flatten())
        return all_y_pred

    def Run(self):
        result = []
        for i in self.featureList:
            result.append(self._run(i))
        mergedArray = [(x,y) for x,y in result if x is not None and y is not None]

        IDs = [x[1] for x in mergedArray]
        mergedArray = [(x,y) for x,y in result if x is not None and y is not None]
        inference_ds = torchDataset_inference(mergedArray)
        inference_dl = DataLoader(inference_ds,
                    batch_size=self.batch_size, worker_init_fn=random_fn,
                    shuffle=False)
        
        y_pred = self.inference(inference_dl)
        res_step2 = pd.DataFrame({'ID':IDs,'y_pred':y_pred})
        res_step2 = pd.DataFrame(res_step2.groupby('ID')['y_pred'].apply(list))
        return res_step2.reset_index()

def stat_ypred(y_preds,THRESHOLD = 0.25):
    if y_preds is np.nan:
        return np.nan
    else:
        pos = [x for x in y_preds if x >= THRESHOLD]
        sig_freq = len(pos)/len(y_preds)
        return round(sig_freq,4)
    
def filterList(featureList,Res):
    res = pd.DataFrame([x for x in featureList if x.ID in Res['ID'].tolist()])
    if len(res) > 0:
        res.columns = feature._fields
        return res
    else:
        return None

def return_res0(final_res,refName,options):
    final_res = final_res.reset_index()
    Loc = [refName + ":" + str(final_res['tpl'][i]+1) + "-" + str(final_res['tpl'][i]+1) + "(+)"
      if final_res['strand'][i] == 1 else
      refName + ":" + str(final_res['tpl'][i]+1) + "-" + str(final_res['tpl'][i]+1) + "(-)"
      for i in range(len(final_res))]
    return_res = pd.DataFrame({'Loc':Loc,'ID': final_res['ID'],
                            'SigQvNum':final_res['sigQ'],
                            'Coverage':final_res['coverage'],
                            'ShufSigQvNum':final_res['shufQ'],
                            'ShufCoverage':final_res['shufCov'],
                            'Freq_ypred':np.nan,
                            'Coverage_ypred':np.nan,
                            'dip.pval':np.nan,
                            'ks2.pval':np.nan,
                            'Type':np.nan})
    return np.array(return_res)

def return_res(final_res,refName,options):
    final_res = final_res.reset_index()
    final_res['y_pred.freq'] = [stat_ypred(x,options.THRESHOLD) for x in final_res['y_pred']]
    final_res['y_pred.coverage'] = [len(x) if x is not np.nan else np.nan for x in final_res['y_pred']]
    Loc = [refName + ":" + str(final_res['tpl'][i]+1) + "-" + str(final_res['tpl'][i]+1) + "(+)"
      if final_res['strand'][i] == 1 else
      refName + ":" + str(final_res['tpl'][i]+1) + "-" + str(final_res['tpl'][i]+1) + "(-)"
      for i in range(len(final_res))]
    return_res = pd.DataFrame({'Loc':Loc,'ID': final_res['ID'],
                            'SigQvNum':final_res['sigQ'],
                            'Coverage':final_res['coverage'],
                            'ShufSigQvNum':final_res['shufQ'],
                            'ShufCoverage':final_res['shufCov'],
                            'Freq_ypred':final_res['y_pred.freq'],
                            'Coverage_ypred':final_res['y_pred.coverage'],
                            'dip.pval':final_res['dipP'],
                            'ks2.pval':final_res['ks2P']})
    
    # return_res['Coverage_ypred'] = return_res['Coverage_ypred'].astype(int)
    return_res['Type'] = ['6mA' if return_res['SigQvNum'][x] >= options.sigQcutoff and return_res['Coverage'][x] >= options.minCoverage_site \
                     and return_res['Freq_ypred'][x] >= options.FREQ_model and return_res['Coverage_ypred'][x] >= options.COVERAGE_model \
                     and return_res['dip.pval'][x] <= options.dipP and return_res['ks2.pval'][x] <= options.ks2P  else np.nan \
                    for x in range(len(return_res))]
    
    return return_res

def get_z_score(array):
    norm = lambda x: (x - np.mean(x))/np.std(x)
    return np.apply_along_axis(norm,1,array)

def preprocess(df,sector=10):
    df['site'] = [x.split("/")[0] for x in df['ID']]
    df['ipd_n_z'] = df['ipdList'].apply(get_z_score)
    df['A'] = df['ipd_n_z'].apply(lambda x: [i[sector] for i in x]).apply(list) 
    df_site = pd.DataFrame(df.groupby('site')['A'].apply(list))
    df_site_dict = df_site.to_dict()['A']
    return df_site_dict

def unlist(a):
    return list(chain(*a))

def combinetest(site,dict1,dict2):
    try:
        dat = np.msort(unlist(dict1[site]))
        ksp = ks_2samp(unlist(dict1[site]),
                    unlist(dict2[site]))
        res = dip.diptst(dat,numt=500)
        return round(res[1],4),round(ksp.pvalue,4)
    except:
        return np.nan,np.nan

#endregion
