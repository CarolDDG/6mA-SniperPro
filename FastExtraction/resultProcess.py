''' Import '''
#region
import warnings
from multiprocessing import Process
from recordtype import recordtype
import pickle
import warnings,logging
log = logging.getLogger(__name__)
__version__ = "1.0"
warnings.filterwarnings("ignore")
infer_feature = recordtype('infer_feature',['ID','kmerList','ipdList','ypred'])
#endregion

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
        log.info("Process %s (PID=%d) started running" % (self.name, self.pid))

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

    def pickleConsumer(self, filename):
        # Open the csv file
        f = open(filename,'wb')
        pickleStream = pickle.Pickler(f)

        try:
            while True:
                n = (yield)
                if n is not None:
                    for x in n:
                        pickleStream.dump(x)
                        pickleStream.clear_memo()
                    log.info("Writing records for {n} bases".format(n = len(n)))
                else:
                    pass

        except GeneratorExit:
            f.close()
            return
        except Exception as e:
            print(e)

    def onStart(self):

        name = self.options.outfile + '.pickle'
        
        if name:
            sink = self.pickleConsumer(name)
            self.sink = sink
        self.sink.send(None)
        
    def onResult(self, resultChunk):
        self.sink.send(resultChunk)

    def onFinish(self):
        self.sink.close()
