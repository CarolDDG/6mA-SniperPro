''' Import '''
import cProfile
import logging
import os.path
from recordtype import recordtype
from multiprocessing import Process
from multiprocessing.process import current_process
import numpy as np
import utils 

''' pre-define. '''
infer_feature = recordtype('infer_feature',['ID','kmerList','ipdList'])

''' Func '''
class WorkerProcess(Process):

    def __init__(self,
                 options,
                 workQueue,
                 resultQueue,
                 sharedCaseAlignmentSet,
                 siteArray):
        Process.__init__(self)
        self.options = options
        self._workQueue = workQueue
        self._resultQueue = resultQueue
        self.sharedCaseAlignmentSet = sharedCaseAlignmentSet
        self.siteArray = siteArray

    def _run(self):
        logging.info("Worker %s (PID=%d) started running" %
                     (self.name, self.pid))
        
        self.caseAlignments = utils._reopen(self.sharedCaseAlignmentSet)
        
        if self.options.randomSeed is None:
            np.random.seed(777)
        else:
            np.random.seed(self.options.randomSeed)
            
        self.onStart()

        while True:
            if self.isTerminated():
                break

            chunkDesc = self._workQueue.get()

            if chunkDesc is None:
                self._resultQueue.put(None)
                self._workQueue.task_done()
                break
            else:
                (chunkId, datum) = chunkDesc
                logging.info("Got chunk: (%s, %s) -- Process: %s" %
                             (chunkId, str(datum), current_process()))
                result = self.onChunk(datum)

                logging.debug("Process %s: putting result." %
                              current_process())
                self._resultQueue.put((chunkId, result))
                self._workQueue.task_done()

        self.onFinish()

        logging.info("Process %s (PID=%d) done; exiting." %
                     (self.name, self.pid))

    def run(self):
        # Make the workers run with lower priority -- hopefully the results writer will win
        # It is single threaded so it could become the bottleneck
        self._lowPriority()
        self._run()

    # ==
    # Begin overridable interface
    # ==
    def onStart(self):
        pass

    def onChunk(self, target):
        """
        This function is the heart of the matter.

        referenceWindow, alnHits -> result
        """
        return None

    def onFinish(self):
        pass

    def isTerminated(self):
        return False

    def _lowPriority(self):
        """
        Set the priority of the process to below-normal.
        """
        os.nice(10)
            
class SniperWorkerProcess(WorkerProcess):

    """
    Manages the summarization of pulse features over a single reference
    """

    def __init__(self,
                 options,
                 workQueue,
                 resultQueue,
                 sharedCaseAlignmentSet,
                 siteArray):
        WorkerProcess.__init__(self, options,workQueue,
                               resultQueue,sharedCaseAlignmentSet,
                              siteArray)
        self.debug = False

    def _prepForReferenceWindow(self, referenceWindow):
        """ Set up member variable to call modifications on a window. """
        self.start = referenceWindow.start
        self.end = referenceWindow.end
        self.refId = referenceWindow.refId
        self.refName = referenceWindow.refName
        self.targetBound = (self.start,self.end)
        
    def hit_preprocess(self):

        alignmentSet = self.sharedCaseAlignmentSet
        
        MIN_IDENTITY = 0.0
        MIN_READLENGTH = 50
        
        hits = [hit for hit in alignmentSet.readsInRange(self.refId,self.start,self.end)
            if ((hit.mapQV >= self.options.mapQvThreshold) and
                (hit.identity >= MIN_IDENTITY) and
                (hit.readLength >= MIN_READLENGTH))]

        hits_pool = {}

        if hits is not None:

            if len(hits) > self.options.maxAlignments:
                hits = np.random.choice(
                    hits, size=self.options.maxAlignments, replace=False)
                
            for hit in hits:
                if hit.HoleNumber in hits_pool.keys():
                    hits_pool[hit.HoleNumber].append(hit)
                else:
                    hits_pool[hit.HoleNumber] = [hit]
            
            hits_pool = utils.postFilter(hits_pool,self.options.minCoverage)

            return hits_pool
        
        else:
            return None
    
    def getInfoList(self,hits,site):

        siteStart = site[1] - self.options.sector
        siteEnd = site[1] + self.options.sector
        ipd_list = []
        read_list = []

        for h in hits:
            try:
                if h.isReverseStrand == site[2]:
                    ipd,read = utils.filterHits(h,siteStart,siteEnd,self.options.sector) 
                    ipd_list.append(ipd.tolist())
                    read_list.append(read.tolist())
            except:
                continue
        
        return ipd_list,read_list 

    def inference_Iteration(self,hits_pool,targetSites):

        featureList = []

        for hole in hits_pool.keys():

            hits = hits_pool[hole]

            intersectSites = utils.siteInHit(targetSites,hits[0])

            if len(intersectSites) == 0:
                pass
            else:
                for site in intersectSites:

                    ID = site[0] + "_" + str(site[2]) + "_" + str(site[1]) + "/" + str(hole)
                    ipd_list,read_list = self.getInfoList(hits,site)
                    result = utils.Run(ipd_list,read_list,self.options.minCoverage,self.options.sector)
                    
                    if result is None:
                        pass
                    else:
                        kmerList,ipdList = result
                        featureList.append(infer_feature(ID = ID,\
                        kmerList = kmerList,
                        ipdList = ipdList))
                                
        return featureList

    def onChunk(self, referenceWindow):
        ''' Get bound region. '''
        self._prepForReferenceWindow(referenceWindow)

        targetBound = (self.refName,self.start,self.end)
       
        hits_pool = self.hit_preprocess()

        if hits_pool is None:
            logging.info("No hits available within this chunk.")
        else:
            targetSites = utils.sitesInRanges(self.siteArray,targetBound)

        chunk_result = self.inference_Iteration(hits_pool,targetSites)
       
        if len(chunk_result) == 0:
            logging.info("No features available within this chunk.")
        else:
            return chunk_result

                