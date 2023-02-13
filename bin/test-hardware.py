#! /usr/bin/env python3
# ---------------------------------------
# This script runs the hardware regression test
# ---------------------------------------
from optparse import OptionParser
import sys, os, time, logging, colorlog, glob, subprocess, multiprocessing, shutil, functools
from multiprocessing import Process, Queue
import maselogger

# Add more test cases here
test_cases = {
    # 'common/join2': [],
    'common/int_mult': [],
    'common/register_slice': [],
    'common/adder_tree_layer': [],
    'common/accumulator': [],
    'common/adder_tree': ['common'],
    'common/vector_mult': ['common'],
    'common/dot_product': ['common'],
    # 'linear/dataflow_linear': ['common'],
    'activations/int_relu': ['common'],
    # 'activations/int_relu6': ['common'],
}


# ---------- TestHardware class --------------
class TestHardware:

    def __init__(self, args, options):
        self.args = args
        self.options = options
        self.isdebug = self.options.debug
        # Root path of mase-tools
        self.root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        self.logger = maselogger.getLogger(
            'test-hardware', os.path.join(self.root, 'test-hardware.log'))
        self.test_cases = self.args
        if self.options.run_all: self.test_cases = test_cases.keys()
        check_fail = self.check_files()
        if check_fail:
            sys.exit(check_fail)

    def check_files(self):
        """Check if all the required files exist for test cases"""
        for test_case in self.test_cases:
            base_name = os.path.basename(test_case)
            # Check if the test bench exists
            if not os.path.exists(
                    os.path.join(self.root, 'hardware', 'testbench', test_case,
                                 '{}_tb.py'.format(base_name))):
                self.logger.error(
                    'Cannot find the testbench! Expected to be {}'.format(
                        os.path.join(self.root, 'hardware', 'testbench',
                                     test_case, '{}_tb.py'.format(base_name))))
                return 1
            # Check if the design file exists
            if not os.path.isfile(
                    os.path.join(self.root, 'hardware',
                                 '{}.sv'.format(test_case))):
                self.logger.error(
                    'Cannot find the design file! Expected to be {}'.format(
                        os.path.join(self.root, 'hardware',
                                     '{}.sv'.format(test_case))))
                return 1
        return 0

    def lint(self):
        """Lint all the required design files"""
        cwd = os.path.join(self.root)
        self.logger.info('Linting all the design files...')
        for test_case in self.test_cases:
            design_file = os.path.join(self.root, 'hardware', test_case)
            include_files = [
                '-I{}'.format(
                    os.path.join(self.root, 'hardware',
                                 '{}'.format(include_file)))
                for include_file in test_cases[test_case]
            ]
            cmd = [
                'verilator', '--lint-only', '--Wall',
                '{}.sv'.format(design_file)
            ] + include_files
            result = self.execute(cmd, log_output=self.isdebug, cwd=cwd)
            if result:
                return 1
        return 0

    def test(self):
        """Test the given test cases"""
        test_count = len(self.test_cases)
        jobs = [None] * test_count
        queue = Queue(test_count)
        for i, test_case in enumerate(self.test_cases):
            jobs[i] = Process(target=self.single_test, args=(test_case, queue))
            jobs[i].start()

        for job in jobs:
            job.join()

        err = 0
        for _ in range(test_count):
            err += queue.get()
        if err:
            self.logger.error(
                'Hardware regression test finished. {} errors.'.format(err))
        else:
            self.logger.info(
                'Hardware regression test finished. {} errors.'.format(err))

        cwd = os.path.join(self.root, 'hardware', 'testbench')
        pycache_files = os.path.join(cwd, '__pycache__')
        if os.path.exists(pycache_files):
            shutil.rmtree(pycache_files)
        return err

    def single_test(self, test_case, queue):
        self.logger.info('Running unit test for {}...'.format(test_case))
        base_name = os.path.basename(test_case)
        cwd = os.path.join(self.root, 'hardware', 'testbench', test_case)
        cmd = ['python3', '{}_tb.py'.format(base_name)]
        result = self.execute(cmd, log_output=self.isdebug, cwd=cwd)
        if result:
            self.logger.error('FAIL.')
        else:
            # Clean files
            pycache_files = os.path.join(cwd, '__pycache__')
            simbuild_files = os.path.join(cwd, 'sim_build')
            result_file = os.path.join(cwd, 'results.xml')
            if os.path.exists(pycache_files):
                shutil.rmtree(pycache_files)
            if os.path.exists(simbuild_files):
                shutil.rmtree(simbuild_files)
            if os.path.isfile(result_file):
                os.remove(result_file)
        queue.put(result)
        return result

    def execute(self, cmd, log_output: bool = True, log_file=None, cwd='.'):
        if log_output:
            self.logger.debug(subprocess.list2cmdline(cmd))
            with subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  bufsize=1,
                                  universal_newlines=True,
                                  cwd=cwd) as result:
                if log_file: f = open(log_file, 'w')
                if result.stdout or result.stderr:
                    self.logger.info('')
                if result.stdout:
                    for line in result.stdout:
                        if log_file: f.write(line)
                        line = line.rstrip("\n")
                        self.logger.trace(line)
                if result.stderr:
                    for line in result.stderr:
                        if log_file: f.write(line)
                        line = line.rstrip("\n")
                        self.logger.trace(line)
                if log_file: f.close()
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=cwd)
        return result.returncode


# ---------- main function --------------
def main():
    USAGE = """Usage: 
test-hardware.py common/int_mult ...
test-hardware.py -a"""

    optparser = OptionParser(usage=USAGE)
    optparser.add_option('-a',
                         '--all',
                         action='store_true',
                         dest='run_all',
                         default=False,
                         help='Run the whole regression test, Default=False')
    optparser.add_option('--debug',
                         action='store_true',
                         dest='debug',
                         default=False,
                         help='Run in debug mode, Default=False')

    (options, args) = optparser.parse_args()
    testrun = TestHardware(args, options)
    lint = testrun.lint()
    if lint: sys.exit(lint)
    run = testrun.test()
    if run: sys.exit(run)
    sys.exit(0)


if __name__ == '__main__':
    main()
