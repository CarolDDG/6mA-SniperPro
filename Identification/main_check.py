''' Import '''
#region
import cProfile
import functools
import os,sys
import logging
import multiprocessing
import threading
import numpy as np
import traceback
from pbcore.io import AlignmentSet
__version__ = "2.1"
from pbcommand.common_options import add_debug_option
from pbcommand.cli import get_default_argparser_with_base_opts, pacbio_args_runner
from pbcommand.utils import setup_log

import utils
from utils import *
from workerProcess import SniperWorkerProcess
from resultProcess import SniperWriter
#endregion

' define '
validateFile = functools.partial(utils._validateResource, os.path.isfile)
validateDir = functools.partial(utils._validateResource, os.path.isdir)

validateNoneOrFile = functools.partial(utils._validateNoneOrResource, os.path.isfile)
validateNoneOrDir = functools.partial(utils._validateNoneOrResource, os.path.isdir)

def get_parser():
    p = get_default_argparser_with_base_opts(
        version=__version__,
        description=__doc__,
        default_level="INFO")
    
    def slurpWindowFile(fname):
        return ",".join(map(str.strip, open(fname).readlines()))

    ''' Parameters must included. ''' 
    #region
    p.add_argument("alignment_set", help="BAM or Alignment DataSet")
    p.add_argument('--control',
                   dest='control',
                   required=True,
                   type=validateNoneOrFile,
                   help='AlignmentSet or mapped BAM file containing a control sample. Tool will perform a case-control analysis')
    p.add_argument("--reference", action="store",
                   required=True,
                   type=validateFile, help="Fasta or Reference DataSet")
    p.add_argument('--outputFile',
                   dest='outfile',
                   required=True,
                   help='Use this option to generate csv output files. Argument here is the root filename of the output files.')
    #endregion
    
    ''' Parameters could be set by default. '''
    #region
    p.add_argument('--outputType', '-ot',
                   dest='outType',
                   default='full',
                   type=str,
                   help='Output type can be selected from "full" or "clean" mode, in "full mode", you will get stat. results for each calculated site, while in "clear" mode only sites meet the thershold selected are shown.')
    p.add_argument('--numWorkers', '-j',
                   dest='numWorkers',
                   default=10,
                   type=int,
                   help='Number of thread to use (-1 uses all logical cpus)')
    p.add_argument("--maxAlignments",
                   type=int,
                   default=1e8,
                   help="Maximum number of hits to process per chunk")
    p.add_argument('--sigQcutoff',
                   dest='sigQcutoff',
                   default=3,
                   type=int,
                   help='Cutoff for sig. q-value counts for each sites.')
    p.add_argument('--minCoverage',
                   dest='minCoverage',
                   default=5,
                   type=int,
                   help='Minimum coverage required for a CCS(subreads num.)')
    p.add_argument('--minCoverage_site',
                dest='minCoverage_site',
                default=10,
                type=int,
                help='Minimum coverage required for a site(CCS num.)')
    p.add_argument("-w", "--referenceWindow", "--referenceWindows",
                   "--refContigs",  # backwards compatibility
                   type=str,
                   dest='referenceWindowsAsString',
                   default=None,
                   help="The window (or multiple comma-delimited windows) of the reference to " + \
                   "be processed, in the format refGroup:refStart-refEnd, split by comma" + \
                   "(default: entire reference).") 
    p.add_argument("-W", "--referenceWindowsFile",
                   "--refContigsFile",  # backwards compatibility
                   type=slurpWindowFile,
                   dest='referenceWindowsAsString',
                   default=None,
                   help="A file containing reference window designations, one per line")   
    p.add_argument("--seed",
                   action="store",
                   dest="randomSeed",
                   type=int,
                   default=777,
                   help="Random seed (for development and debugging purposes only),ensure that results can be reproduced, 777 as default")
    p.add_argument("--referenceStride", action="store", type=int,
                   default=50000,
                   help="Size of reference window in internal " +
                   "parallelization.  For testing purposes only.")
    p.add_argument("--sector",
                   type=int,
                   default=10,
                   help="Range for expanding(both up/down streams).")
    p.add_argument('--maxQueueSize',
                   dest='maxQueueSize',
                   default=100000,
                   type=int,
                   help='Max Queue Size.')
    p.add_argument('--modelDict',
                   dest='modelDict',
                   default="/home/user/data3/wangjx/prj_007_6mA/myModel/new_240304/train_dataset/New_240314_bbasic/Mouse/new_4318_whole_freq0.5/train_results/kfold2_model.0521.2010.pt",
                   help='State_dict for predicting model.')
    p.add_argument('--modelPath',
                dest='modelPath',
                default="/home/user/data3/wangjx/prj_007_6mA/Arrange_allData_240529/Step2ModelTrainingAndInference/model_training_240603/model",
                help='Model stored path.')
    p.add_argument('--COVERAGE_model',
                dest='COVERAGE_model',
                default=10,
                type=int,
                help='Minimum coverage for model prediction(site-level).')
    p.add_argument("--FREQ_model",
                   dest="FREQ_model",
                   type=float,
                   default=0.4,
                   help="Freq. cutoff for model predictions(site-level).")
    p.add_argument('--THRESHOLD_model',
                   dest='THRESHOLD',
                   default=0.25,
                   type=float,
                   help='Threshold cutoff for model predictions(site-level).')
    p.add_argument('--dip_P',
                dest='dipP',
                default=0.5,
                type=float,
                help='Threshold cutoff for dip test p-val.')
    p.add_argument('--ks2_P',
            dest='ks2P',
            default=0.5,
            type=float,
            help='Threshold cutoff for ks 2 sample test p-val.')
    #endregion
    
    ''' Backup parameters '''
    #region
    # p.add_argument('--mapQvThreshold',
    #             dest='mapQvThreshold',
    #             type=float,
    #             default=20.0)
    # p.add_argument("--num",
    #             default=5,
    #             type=int,
    #             help="Number of subreads you want to preserve for each site while doing model preprocess calculation.")
    # p.add_argument("--maxLength",
    #                type=int,
    #                default=1e6,
    #                help="Maximum number of bases to process per contig")

     # p.add_argument('--maxCoverage',
    #                dest='maxCoverage',
    #                type=int, default=-1,
    #                help='Maximum coverage to use at each site')
        # p.add_argument("--alignmentSetRefWindows",
    #             action="store_true",
    #             dest="referenceWindowsFromAlignment",
    #             help="Use refWindows in dataset")
    # p.add_argument("--profile",
    #             action="store_true",
    #             dest="doProfiling",
    #             default=False,
    #             help="Enable Python-level profiling (using cProfile).")
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
        # Work chunks are created by the main thread and put on this queue
        # They will be consumed by KineticWorker threads, stored in
        # self._workers
        self._workQueue = multiprocessing.JoinableQueue(
            self.options.maxQueueSize)
        # Completed chunks are put on this queue by KineticWorker threads
        # They are consumed by the KineticsWriter process
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
                self.controlAlignments)
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
        self.controlAlignments = AlignmentSet(self.args.control,
                               referenceFastaFname=self.args.reference)

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
        else:
            self.referenceWindows = utils.createReferenceWindows(
                self.refInfo)

        # Spawn workers
        self._launchSlaveProcesses()

        logging.info(
            "Generating sniper's summary for [%s]" % self.args.alignment_set)

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
        logging.info("Main process finished. Exiting.")
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
