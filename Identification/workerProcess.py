''' Import '''
from multiprocessing import Process
from multiprocessing.process import current_process
import utils
from recordtype import recordtype
from utils import modelInference
import pandas as pd
import logging
import numpy as np
import sys,os

''' predefine. '''
#region
ipdRec = [('tpl','<u4'),
        ('zmwhole','<u4'),
        ('strand','<i8'),
        ('rv_ipd','<f4')]
resRec = [('tpl','<u4'),
        ('strand','<i8'),
          ('sigQ','<u4'),
        ('coverage','<u4'),
         ('shufQ','<u4'),
         ('shufCov','<u4')]
feature = recordtype('feature',['ID',
'kmerList','ipdList'])

''' Func '''
class WorkerProcess(Process):

    def __init__(self,options,workQueue, resultQueue,
                 sharedCaseAlignmentSet,sharedControlAlignmentSet):
        Process.__init__(self)
        self.options = options
        self._workQueue = workQueue
        self._resultQueue = resultQueue
        self.sharedCaseAlignmentSet = sharedCaseAlignmentSet
        self.sharedControlAlignmentSet = sharedControlAlignmentSet

    def _run(self):
        logging.info("Worker %s (PID=%d) started running" %
                     (self.name, self.pid))
        
        self.caseAlignments = utils._reopen(self.sharedCaseAlignmentSet)

        self.controlAlignments = utils._reopen(self.sharedControlAlignmentSet)
        
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
                # Sentinel indicating end of input.  Place a sentinel
                # on the results queue and end this worker process.
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

    def onStart(self):
        pass

    def onChunk(self):
        return None

    def onFinish(self):
        pass

    def isTerminated(self):
        return False

    def _lowPriority(self):
        os.nice(10)
            
class SniperWorkerProcess(WorkerProcess):

    """
    Main running process.
    """

    def __init__(self,
                 options,
                 workQueue,
                 resultQueue,
                 sharedCaseAlignmentSet,
                 sharedControlAlignmentSet):
        WorkerProcess.__init__(self, options,workQueue,
                               resultQueue, sharedCaseAlignmentSet,
                              sharedControlAlignmentSet)
        self.debug = False

    def _prepForReferenceWindow(self, referenceWindow):
        """ Set up member variable to call modifications on a window. """
        self.start = referenceWindow.start
        self.end = referenceWindow.end
        # FIXME some inconsistency in how reference info is retrieved -
        # DataSet API uses Name, ipdModel.py uses ID
        self.refId = referenceWindow.refId
        self.refName = referenceWindow.refName
        # refInfoTable = self.caseAlignments.referenceInfo(self.refName)
        self.targetBound = (self.start,self.end)
        
    def hit_preprocess(self,type='case'):

        if type == "case":
            alignmentSet = self.sharedCaseAlignmentSet
        else:
            alignmentSet = self.sharedControlAlignmentSet
        
        MAPQV = 20
        MIN_IDENTITY = 0.0
        MIN_READLENGTH = 50

        hits = [hit for hit in alignmentSet.readsInRange(self.refId,self.start,self.end)
            if ((hit.mapQV >= MAPQV) and
                (hit.identity >= MIN_IDENTITY) and
                (hit.readLength >= MIN_READLENGTH))]

        if hits is not None:
            if len(hits) > self.options.maxAlignments:
                hits = np.random.choice(hits, size=self.options.maxAlignments, replace=False)
            
            raw_hits_pool = {}
            rv_hits_pool = np.array([],dtype = ipdRec)
             
            for hit in hits:
                rv_hits_pool = np.concatenate([rv_hits_pool,utils.RV_for_hit(hit,self.targetBound)])
                if hit.HoleNumber in raw_hits_pool.keys():
                        raw_hits_pool[hit.HoleNumber].append(hit)
                else:
                        raw_hits_pool[hit.HoleNumber] = [hit]           
            raw_hits_pool = utils.postFilter(raw_hits_pool,self.options.minCoverage) 
            return raw_hits_pool,rv_hits_pool
        else:
            return None  
    
    def step1Test(self,rv_hits_pool,rv_chits_pool):
        rv_hits_merged = utils.caseMerged(rv_hits_pool)
        rv_chits_merged = utils.controlMerged(rv_chits_pool)
        rv_hits_merged = utils.postFilter(rv_hits_merged,self.options.minCoverage)
        rv_chits_merged = utils.postFilter(rv_chits_merged,self.options.minCoverage)
        
        logging.info("Retrieved %d filtered case hits" % (len(rv_hits_merged)))
        logging.info("Retrieved %d filtered control hits" % (len(rv_chits_merged)))

        if len(rv_hits_merged) > 0 and len(rv_chits_merged) > 0:
            try:
                result = utils.doTest(rv_hits_merged,rv_chits_merged)
                res_dict_p,res_dict_fc = utils.resMerged(result)
                res_dict_p = utils.postFilter(res_dict_p,self.options.minCoverage_site)
                res_dict_fc = utils.postFilter(res_dict_fc,self.options.minCoverage_site)
                res_dict_q =  {key:utils.correct_pvalues_for_multiple_testing(value) for key,value in res_dict_p.items()}
                res_dict_shufQ,res_dict_shufFC = utils.shuffleFCandQ(res_dict_p,res_dict_fc,self.options.minCoverage_site)
                assert len(res_dict_p) == len(res_dict_fc) == len(res_dict_q) == len(res_dict_shufQ) == len(res_dict_shufFC)
                res_step1,candiArray = utils.returnStep1Res(res_dict_q,res_dict_fc,res_dict_shufQ,
                res_dict_shufFC,self.options.minCoverage_site,self.options.sigQcutoff)
                res_step1['ID'] = [self.refName + "_" + str(res_step1['strand'][x]) + "_" + str(res_step1['tpl'][x]) for x in range(len(res_step1))]
                return res_step1,candiArray
            except:
                logging.info("Error in doing test in range:" + self.refName + ":" + str(self.start) + "-" + str(self.end))
                return None

    def step2Extract(self,candiArray,hits_pool):
        featureList = utils.Extract(self.refName,candiArray,hits_pool,self.options.sector,self.options.minCoverage)
        return featureList   

    def step3PostDistFilter(self,caseFeatureList,ctrlFeatureList,res,options):
        caseDf = utils.filterList(caseFeatureList,res)
        ctrlDf = utils.filterList(ctrlFeatureList,res)
        if caseDf is not None and ctrlDf is not None:
            case_site_dict = utils.preprocess(caseDf,options.sector)
            ctrl_site_dict = utils.preprocess(ctrlDf,options.sector)
            resList = []
            for i in case_site_dict.keys():
                res = utils.combinetest(i,case_site_dict,ctrl_site_dict)
                resList.append(res)
            res_step3 = pd.DataFrame(resList)
            res_step3.index = case_site_dict.keys()
            res_step3.columns = ['dipP','ks2P']
            res_step3['ID'] = res_step3.index
            return res_step3
        else:
            return None
        
    def onChunk(self, referenceWindow):
        
        ''' parameters set: can be rewrite -> sys.argv. '''
        byte = np.dtype('byte')
        uint8 = np.dtype('uint8')
        
        ''' Get bound region. '''
        self._prepForReferenceWindow(referenceWindow)

        raw_hits_pool,rv_hits_pool = self.hit_preprocess(type = "case")
        raw_chits_pool,rv_chits_pool = self.hit_preprocess(type = "control")
        
        if raw_hits_pool is not None and rv_hits_pool is not None:
            out = self.step1Test(rv_hits_pool,rv_chits_pool)
            if out is not None:
                res_step1,candiArray1 = out
                if len(candiArray1) > 0:
                    caseFeatureList1 = self.step2Extract(candiArray1,raw_hits_pool)
                    ctrlFeatureList1 = self.step2Extract(candiArray1,raw_chits_pool)
                    modelInfer = modelInference(caseFeatureList1,self.options.modelDict,
                                                    self.options.modelPath)
                    if modelInfer is not None:
                        res_step2 = modelInfer.Run()
                        if len(res_step2) > 0 and len(ctrlFeatureList1) > 0:
                            res_step3 = self.step3PostDistFilter(caseFeatureList1,ctrlFeatureList1,res_step2,self.options)
                            if res_step3 is not None:
                                merge1 = pd.merge(res_step1,res_step2,on = 'ID',how = 'left')
                                merge2 = pd.merge(merge1,res_step3,on = 'ID',how = 'left')
                                final_res = utils.return_res(merge2,self.refName,self.options)
                                if self.options.outType == "full":
                                    return np.array(final_res)
                                else:
                                    return np.array(final_res[final_res['Type'] == '6mA'])
                            elif self.options.outType == "full":
                                final_res = utils.return_res0(res_step1,self.refName,self.options)
                                return final_res
                            else:
                                return None
                        elif self.options.outType == "full":
                                final_res = utils.return_res0(res_step1,self.refName,self.options)
                                return final_res
                        else:
                            return None
                    elif self.options.outType == "full":
                        final_res = utils.return_res0(res_step1,self.refName,self.options)
                        return final_res
                    else:
                        return None
                elif len(res_step1)>0 and self.options.outType == "full":
                    final_res = utils.return_res0(res_step1,self.refName,self.options)
                    return final_res
                else:
                    return None
            else:
                return None
        else:
            return None
