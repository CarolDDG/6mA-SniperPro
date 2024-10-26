''' Import '''
import logging
import warnings
from multiprocessing import Process
warnings.filterwarnings("ignore")
__version__ = "2.1"
log = logging.getLogger(__name__)
import numpy as np
from recordtype import recordtype
import pickle

''' Func. '''    
class ResultCollectorProcess(Process):

    """
    Gathers results and writes to a file.
    """

    def __init__(self, options, resultQueue):
        Process.__init__(self)
        self.daemon = True
        self.options = options
        self._resultQueue = resultQueue

    def _run(self):
        log.info("Result collection process %s (PID=%d) started running" % (self.name, self.pid))

        self.onStart()

        nextChunkId = 0
        chunkCache = {}

        sentinelsReceived = 0
        
        while sentinelsReceived < self.options.numWorkers:
            result = self._resultQueue.get()
            self._resultQueue.task_done()

            if result is None:
                sentinelsReceived += 1
            else:
                # Write out chunks in chunkId order.
                # Buffer received chunks until they can be written in order
                (chunkId, datum) = result
                chunkCache[chunkId] = datum
               
                while nextChunkId in chunkCache:
                    nextChunk = chunkCache.pop(nextChunkId)
                    self.onResult(nextChunk)
                    nextChunkId += 1

        log.info("Result thread shutting down...")
        self.onFinish()

    def run(self):
        self._run()

    # ==================================
    # Overridable interface begins here.
    #
    def onStart(self):
        pass

    def onResult(self, result):
        pass

    def onFinish(self):
        pass

class SniperWriter(ResultCollectorProcess):
    
    ''' Model inference can be added here. '''
    def __init__(self, options,resultQueue, refInfo):
        ResultCollectorProcess.__init__(self,options,resultQueue)
        self.refInfo = refInfo
        
    def openWriteHandle(self, filename):
        fileobj = open(filename, "w", 2 << 17)
        return fileobj

    def resConsumer(self, filename):
        """
        Consume summary rows and write them to tsv
        """
    
        name1 = self.options.outfile + '.stat.tsv'
       
        # Open the csv file
        f1 = self.openWriteHandle(name1)
        
        delim = "\t"

        # print header
        header = ['Loc','ID','SigQvNum_stat','Coverage_stat','ShufSigQvNum_stat','Coverage_shufStat',
                  'Freq_ypred','Coverage_ypred','dip.pval','ks2.pval','Type']
        print(delim.join(header),file = f1)

        # Special cases for formatting columns of the csv
        try:
            while True:
            # Pull a list of record in from the producer
                itemList = (yield)
                if itemList is not None and len(itemList)>0:
                    return_res = itemList
                    for i in return_res:
                        stri = [str(x) for x in i]
                        print(delim.join(stri),file = f1)
                    log.info("Writing stat records for {n} bases".format(n = len(return_res)))
                else:
                    pass
        except GeneratorExit:
            f1.close()
            return
        except Exception as e:
            print(e)

    def onStart(self):

        # Spec for what kinds of output files we can generate.
        # Entry format is (<option field name>, <extension>, <writer consumer
        # function>)
        name = self.options.outfile 
        
        if name:
            sink = self.resConsumer(name)
            self.sink = sink
        self.sink.send(None)
        
    def onResult(self, resultChunk):
        self.sink.send(resultChunk)

    def onFinish(self):
        self.sink.close()
