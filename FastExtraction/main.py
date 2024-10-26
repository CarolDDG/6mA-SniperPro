''' Import '''
#region
__version__ = "3.0"
import cProfile
import functools
import os
import csv
import logging
import sys
import multiprocessing
import threading
import numpy as np
import traceback
from pbcore.io import AlignmentSet
from pbcommand.cli import get_default_argparser_with_base_opts, pacbio_args_runner
from pbcommand.utils import setup_log
from workerProcess import SniperWorkerProcess
from resultProcess import SniperWriter
import utils
from utils import *
#endregion

''' Get the argument '''
#region
validateFile = functools.partial(utils._validateResource, os.path.isfile)
validateDir = functools.partial(utils._validateResource, os.path.isdir)

validateNoneOrFile = functools.partial(utils._validateNoneOrResource, os.path.isfile)
validateNoneOrDir = functools.partial(utils._validateNoneOrResource, os.path.isdir)

def get_parser():
    p = get_default_argparser_with_base_opts(
        version=__version__,
        description=__doc__,
        default_level="INFO")
    ''' Parameters must included. ''' 
    #region
    p.add_argument("alignment_set", help="BAM or Alignment DataSet")
    p.add_argument('--sitesFile',
                   dest='siteFilename',
                   required=True,
                   type=validateNoneOrFile,
                   help='Candidate 6mA sites list identified by step1.')
    p.add_argument("--reference", action="store",
                   required=True,
                   type=validateFile, help="Fasta or Reference DataSet.")
    p.add_argument('--outputFile',
                   dest='outfile',
                   required=True,
                   help='Use this option to generate csv output files. Argument here is the root filename of the output files.')
    #endregion
    
    ''' Parameters could be set by default. '''
    #region
    p.add_argument('--numWorkers', '-j',
                   dest='numWorkers',
                   default=5,
                   type=int,
                   help='Number of thread to use (-1 uses all logical cpus), DEFAULT:5.')
    p.add_argument("--sector",
                   type=int,
                   default=10,
                   help="Range for expanding(both up/down streams), DEAFULT:10.")
    p.add_argument("--maxAlignments",
                   type=int,
                   default=100000000,
                   help="Maximum number of hits to process per chunk, DEFAULT: 1e8")
    p.add_argument('--minCoverage',
                   dest='minCoverage',
                   default=5,
                   type=int,
                   help='Minimum coverage required to call a modified base (CCS level), DEFAULT:5')
    p.add_argument('--maxQueueSize',
                   dest='maxQueueSize',
                   default=100000,
                   type=int,
                   help='Max Queue Size, DEFAULT: 1e5')
    p.add_argument('--mapQvThreshold',
                   dest='mapQvThreshold',
                   type=float,
                   default=20.0)
    p.add_argument("-w", "--referenceWindow", "--referenceWindows",
                   "--refContigs",  # backwards compatibility
                   type=str,
                   dest='referenceWindowsAsString',
                   default=None,
                   help="The window (or multiple comma-delimited windows) of the reference to " + \
                   "be processed, in the format refGroup:refStart-refEnd" + \
                   "(DEFALUT: entire reference).")
    def slurpWindowFile(fname):
        return ",".join(map(str.strip, open(fname).readlines()))
    p.add_argument("-W", "--referenceWindowsFile",
                   "--refContigsFile",  # backwards compatibility
                   type=slurpWindowFile,
                   dest='referenceWindowsAsString',
                   default=None,
                   help="A file containing reference window designations, one per line")
    p.add_argument("--alignmentSetRefWindows",
                action="store_true",
                dest="referenceWindowsFromAlignment",
                help="Use refWindows in dataset.")
    p.add_argument("--seed",
                   action="store",
                   dest="randomSeed",
                   type=int,
                   default=777,
                   help="Random seed, DEFAULT:777")
    p.add_argument("--referenceStride", action="store", type=int,
                   default=50000,
                   help="Size of reference window in internal parallelization, DEFAULT: 50000.")
    #endregion
    
    ''' Backup parameters '''
    #region
    # p.add_argument("--profile",
    #                action="store_true",
    #                dest="doProfiling",
    #                default=False,
    #                help="Enable Python-level profiling (using cProfile).")
    # p.add_argument('--maxCoverage',
    #             dest='maxCoverage',
    #             type=int, default=-1,
    #             help='Maximum coverage to use at each site.')
    # p.add_argument("--maxLength",
    #                type=int,
    #                default=1e6,
    #                help="Maximum number of bases to process per contig")
    #endregion
    return p

class SniperRunner(object):
    def __init__(self, args):
        self.args = args
        self.alignments = None

    def start(self):
        self.validateArgs()
        return self.run()

    def getVersion(self):
        return __version__

    def validateArgs(self):
        parser = get_parser()
        if not os.path.exists(self.args.alignment_set):
            parser.error('Input AlignmentSet file provided does not exist')

    def run(self):
        
        self.options = self.args
        self.options.cmdLine = " ".join(sys.argv)
        self._workers = []

        # set random seed
        # XXX note that this is *not* guaranteed to yield reproducible results
        # indepenently of the number of processing cores used!
        if self.options.randomSeed is not None:
            np.random.seed(self.options.randomSeed)

        try:
            ret = self._mainLoop()
        finally:
            # Be sure to shutdown child processes if we get an exception on
            # the main thread
            for w in self._workers:
                if w.is_alive():
                    w.terminate()

        return ret

    def _initQueues(self):
        self._workQueue = multiprocessing.JoinableQueue(
            self.options.maxQueueSize)
        self._resultQueue = multiprocessing.JoinableQueue(
            self.options.maxQueueSize)

    def _launchSlaveProcesses(self):
        availableCpus = multiprocessing.cpu_count()
        logging.info("Available CPUs: %d" % (availableCpus,))
        logging.info("Requested worker processes: %d" %
                     (self.options.numWorkers,))

        # Use all CPUs if numWorkers < 1
        if self.options.numWorkers < 1:
            self.options.numWorkers = availableCpus

        # Warn if we make a bad numWorker argument is used
        if self.options.numWorkers > availableCpus:
            logging.warn("More worker processes requested (%d) than CPUs available (%d);"
                         " may result in suboptimal performance."
                         % (self.options.numWorkers, availableCpus))

        self._initQueues()

        # Launch the worker processes
        self._workers = []
        for i in range(self.options.numWorkers):
            p = SniperWorkerProcess(
                self.options,
                self._workQueue,
                self._resultQueue,
                self.caseAlignments,
                self.siteArray)
            self._workers.append(p)
            p.start()
        logging.info("Launched worker processes.")

        # Launch result collector    
        self._resultCollectorProcess = SniperWriter(self.options,self._resultQueue,self.refInfo)
        self._resultCollectorProcess.start()
        logging.info("Launched result collector process.")

        # Spawn a thread that monitors worker threads for crashes
        self.monitoringThread = threading.Thread(target=utils.monitorChildProcesses, args=(
            self._workers + [self._resultCollectorProcess],))
        self.monitoringThread.start()

    def loadSharedSitesArray_infer(self,siteFilename):
        logging.info("Reading sites' file: %s" % siteFilename)
        self.siteFile = siteFilename
        resultRec = [('refName','O'),('tpl','<u4'),\
             ('strand','<u2')]
        n = 0
        with open(self.siteFile,'r') as f:
            reader = csv.reader(f)
            refName = []
            tpl = []
            strand = []
            for line in reader:
                refName.append(line[0])
                tpl.append(int(line[1]))
                strand.append(line[2])
            sitesArray = np.zeros(len(refName),dtype = resultRec)
            sitesArray['refName'] = refName
            sitesArray['tpl'] = tpl
            sitesArray['strand'] = strand
        
        self.siteArray = sitesArray
        f.close()

    def loadSharedAlignmentSet(self, alignmentFilename):
        """
        Read the input AlignmentSet so the indices can be shared with the
        slaves.  This is also used to pass to ReferenceUtils for setting up
        the ipdModel object.
        """
        logging.info("Reading AlignmentSet: %s" % alignmentFilename)
        logging.info("           reference: %s" % self.args.reference)
        self.caseAlignments = self.alignments = AlignmentSet(alignmentFilename,
                                       referenceFastaFname=self.args.reference)
        # XXX this should ensure that the file(s) get opened, including any
        # .pbi indices - but need to confirm this
        self.refInfo = self.alignments.referenceInfoTable

    def _mainLoop(self):
     
        self.loadSharedAlignmentSet(self.args.alignment_set)
        self.loadSharedSitesArray_infer(self.args.siteFilename)

        # Resolve the windows that will be visited.
        if self.args.referenceWindowsAsString is not None:
            self.referenceWindows = []
            for s in self.args.referenceWindowsAsString.split(","):
                try:
                    win = utils.parseReferenceWindow(
                        s, self.alignments.referenceInfo)
                    self.referenceWindows.append(win)
                except BaseException:
                    continue
        elif self.args.referenceWindowsFromAlignment:
            self.referenceWindows = utils.referenceWindowsFromAlignment(
                self.alignments, self.alignments.referenceInfo)
            refNames = set([rw.refName for rw in self.referenceWindows])
            self.refInfo = [r for r in self.refInfo if r.Name in refNames]
        else:
            self.referenceWindows = utils.createReferenceWindows(
                self.refInfo)

        # Spawn workers
        self._launchSlaveProcesses()

        logging.info(
            'Generating snipers summary for [%s]' % self.args.alignment_set)

        self.workChunkCounter = 0

        # Iterate over references
        for window in self.referenceWindows:
            logging.info('Processing window/contig: %s' % (window,))
            for chunk in utils.enumerateChunks(
                    self.args.referenceStride, window):
                self._workQueue.put((self.workChunkCounter, chunk))
                self.workChunkCounter += 1

        # Shutdown worker threads with None sentinels
        for i in range(self.args.numWorkers):
            self._workQueue.put(None)

        for w in self._workers:
            w.join()

        # Join on the result queue and the resultsCollector process.
        # This ensures all the results are written before shutdown.
        self.monitoringThread.join()
        self._resultQueue.join()
        self._resultCollectorProcess.join()
        logging.info("main.py finished. Exiting.")
        self.alignments.close()
        return 0

def args_runner(args):
    kt = SniperRunner(args)
    return kt.start()

#endregion

''' Set the main func. '''
def main(argv=sys.argv, out=sys.stdout):

    try:
        return pacbio_args_runner(
            argv=argv[1:],
            parser=get_parser(),
            args_runner_func=args_runner,
            alog=logging.getLogger(__name__),
            setup_log_func=setup_log)
    # FIXME is there a more central place to deal with this?
    except Exception as e:
        type, value, tb = sys.exc_info()
        traceback.print_exc(file=sys.stderr)
        # Note: if kt.args.usePdb
        # This won't work. If an exception is raised in parseArgs,
        # then kt.args is not defined yet.
        if '--pdb' in argv:
            try:
                # this has better integration with ipython and is nicer
                # pip install ipdb
                import ipdb
                ipdb.post_mortem(tb)
            except ImportError:
                import pdb
                pdb.post_mortem(tb)
        else:
            # exit non-zero
            raise

if __name__ == "__main__":
    sys.exit(main())
