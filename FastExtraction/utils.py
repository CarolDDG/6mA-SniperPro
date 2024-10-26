''' Import '''
from pbcore.io import ReferenceSet
from collections import namedtuple
from urllib.parse import urlparse
import cProfile
import threading
import random
import multiprocessing
from pbcore.io import AlignmentSet
import itertools
import logging
from multiprocessing.sharedctypes import RawArray
import warnings
import numpy as np
import os,sys,gzip
import time
from multiprocessing import Process
import math
import copy
from pbcore.io.opener import (openAlignmentFile, openIndexedAlignmentFile)
from multiprocessing.process import current_process
from collections import namedtuple, defaultdict
import cProfile
import logging
from multiprocessing import Process
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as stats
import re 
import pandas as pd
import numpy as np
import scipy.stats as stats
import re 
import torch
from torch.utils.data import Dataset
import datetime
from torch.utils.data import DataLoader
from collections import OrderedDict
from itertools import chain
from torch.utils.data._utils.collate import default_collate

''' predefine. '''
ReferenceWindow = namedtuple(
    "ReferenceWindow", ["refId", "refName", "start", "end"])

''' Func. '''   
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

def _reopen(self):
    """
    Force re-opening of underlying alignment files, preserving the
    reference and indices if present, and return a copy of the
    AlignmentSet.  This is a workaround to allow us to share the index
    file(s) already loaded in memory while avoiding multiprocessing
    problems related to .bam files.
    """
    refFile = self._referenceFile
    newSet = copy.deepcopy(self)
    newSet._referenceFastaFname = refFile
    indices = [f.index for f in self.resourceReaders()]
    self.close()
    _openFiles(newSet, refFile=refFile, sharedIndices=indices)
    return newSet
         
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

def Hit_trimming(hit,targetBound):
    start,end = targetBound
    thit = hit.clippedTo(start,end)
    return thit

def postFilter(hits_dict,cutoff):
    for i in list(hits_dict):
        if len(hits_dict[i]) < cutoff:
            del hits_dict[i]
    return hits_dict

def siteInHit(sitesArray,hit):
    intersect = np.in1d(sitesArray['tpl'],\
                        hit.referencePositions(),\
                        assume_unique = True)
    return sitesArray[intersect]

def filterHits(aln,start,end,sector=10):
    matched = np.array([x != '-' for x in aln.read()])
    np.logical_and(np.logical_not(np.isnan(aln.IPD())),matched,out = matched)
    np.logical_and(np.logical_not(np.isnan(aln.PulseWidth())),matched,out = matched)
    uniq_bool = matched
    referencePositions = aln.referencePositions()
    np.logical_and(referencePositions <= end, uniq_bool, uniq_bool)
    np.logical_and(referencePositions >= start, uniq_bool, uniq_bool)
    nm = uniq_bool.sum()
    if nm != 2*sector+1:
        return None
    else:
        ipd = aln.IPD()[uniq_bool]
        read = coden_complement(list(aln.read()))[uniq_bool]
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

def sitesInRanges(array,targetBound):
    CHR,start,end = targetBound
    ''' left reach, right unreach. '''
    array = array[array['refName'] == CHR]
    new = array[(array['tpl'] >= start) & (array['tpl'] < end)]
    return new
   
def update(ipd_list,read_list):
    read_list = np.array(read_list).T
    read_list_update = np.array([mode1(col) for col in read_list]) 
    return np.array(ipd_list),read_list_update
    
def Run(ipd_list,read_list,MIN_COVERAGE,sector = 10):

    ipd_list_update,read_list_update = update(ipd_list,read_list)

    if len(ipd_list_update) >= MIN_COVERAGE:
        idx = [x for x in range(len(ipd_list_update))]
        np.random.shuffle(idx)
        ipd_list_update = ipd_list_update[idx[0:MIN_COVERAGE],:]
        ki = read_list_update,ipd_list_update
        if ki is not None:
            kmerList,ipdList = ki
            return kmerList,ipdList
        else:
            return None
    else:
        return None
