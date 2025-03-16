import logging, torch
from pathlib import Path
from textwrap import indent

from chop.passes.graph.utils import vf, v2p, init_project
from chop.nn.quantizers import (
    integer_quantizer_for_hw,
    integer_quantizer,
)

logger = logging.getLogger(__name__)

from pathlib import Path

torch.manual_seed(0)

import cocotb
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
# from mase_cocotb.utils import  sign_extend, signed_to_unsigned


import dill
import inspect


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def _emit_cocotb_test(graph, pass_args={}):

    wait_time = pass_args.get("wait_time", 2)
    wait_unit = pass_args.get("wait_units", "ms")
    batch_size = pass_args.get("batch_size", 1)

    test_template = f"""
import cocotb

@cocotb.test()
async def test(dut):
    from pathlib import Path
    import dill
    from cocotb.triggers import Timer

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    with open(tb_path / "tb_obj.dill", "rb") as f:
        tb = dill.load(f)(dut, fail_on_checks=True)

    await tb.initialize()

    in_tensors = tb.generate_inputs(batches={batch_size})
    exp_out = tb.model(*list(in_tensors.values()))

    # print ("in_tensors: ", in_tensors)
    # print (tb.model)
    # print ("weight: ", tb.model.fc1.weight)
    # print ("Bias: ", tb.model.fc1.bias)
    # print ("exp_out: ", exp_out)

    tb.load_drivers(in_tensors)
    tb.load_monitors(exp_out)

    def bin_to_signed_int(binary_string):
        bits = len(binary_string)
        value = int(binary_string, 2)
        # If the sign bit is set, convert using two's complement
        if value >= (1 << (bits - 1)):
            value -= (1 << bits)
        return value

    # Monitor DUT during execution
    for cycle in range(700):  # Observe for 50 cycles
        await Timer(20, units="ns")  # Adjust timing based on design
        # conv_in = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.data_in_0.value]
        # reshape_in = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.data_in_reshaper_0.data_in.value]
        # reshape_out = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.data_in_reshaper_0.data_out.value]
        # pad_in = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.padding_mase_inst1.data_in.value]
        # pad_out =[bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.padding_mase_inst1.data_out.value]
        # std_in = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.striding_inst1.pixel_in.value]
        # std_out = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.striding_inst1.result_out.value]
        # conv_arith_in = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.conv_arith_inst.data_in.value]
        # pool_in = [bin_to_signed_int(str(bit)) for bit in dut.maxpool_inst.data_in_0.value]
        # pool_out = [bin_to_signed_int(str(bit)) for bit in dut.maxpool_inst.data_out_0.value]




        # conv_out = [bin_to_signed_int(str(bit)) for bit in dut.conv1_inst.data_out_0.value]

        # tb._log.info(f"Cycle {{cycle+1}}:")
        # tb._log.info(f"  conv_in: {{conv_in}}")
        # tb._log.info(f"  reshape_in: {{reshape_in}}")
        # tb._log.info(f"  reshape_out: {{reshape_out}}")
        # tb._log.info(f"  pad_in: {{pad_in}}")
        # tb._log.info(f"  pad_out: {{pad_out}}")
        # tb._log.info(f"  std_in: {{std_in}}")
        # tb._log.info(f"  std_out: {{std_out}}")
        # tb._log.info(f"  conv_arith_in: {{conv_arith_in}}")
        # tb._log.info(f"  conv_arith_out: {{dut.conv1_inst.conv_arith_inst.arith_data_out.value}}")
        # tb._log.info(f"  rd_in: {{dut.conv1_inst.fr_inst.in_data.value}}")
        # tb._log.info(f"  rd_out: {{dut.conv1_inst.fr_inst.out_data.value}}")
        # tb._log.info(f"  buff_in: {{dut.conv1_inst.out_buffer_inst.data_in.value}}")
        # tb._log.info(f"  buff_in_valid: {{dut.conv1_inst.out_buffer_inst.data_in_valid.value}}")
        # tb._log.info(f"  buff_in_ready: {{dut.conv1_inst.out_buffer_inst.data_in_ready.value}}")
        # tb._log.info(f"  buff_out: {{dut.conv1_inst.out_buffer_inst.data_out.value}}")
        # tb._log.info(f"  buff_out_valid: {{dut.conv1_inst.out_buffer_inst.data_out_valid.value}}")
        # tb._log.info(f"  conv_out_valid: {{dut.conv1_inst.data_out_0_valid.value}}")
        # tb._log.info(f"  buff_out_ready: {{dut.conv1_inst.out_buffer_inst.data_out_ready.value}}")
        # tb._log.info(f"  out: {{conv_out}}")

        # tb._log.info(f"  in: {{pool_in}}")
        # tb._log.info(f"  max_data_in_0_valid: {{dut.maxpool_inst.data_in_0_valid.value}}")
        # tb._log.info(f"  max_data_in_0_ready: {{dut.maxpool_inst.data_in_0_ready.value}}")
        # tb._log.info(f"  out: {{pool_out}}")

    await tb.wait_end(timeout={wait_time}, timeout_unit="{wait_unit}")
"""

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "test.py", "w") as f:
        f.write(test_template)


def _emit_cocotb_tb(graph):
    class MaseGraphTB(Testbench):
        def __init__(self, dut, fail_on_checks=True):
            super().__init__(dut, dut.clk, dut.rst, fail_on_checks=fail_on_checks)
                    # Initialize a logger for the testbench
            self._log = logging.getLogger("MaseGraphTB")
            self._log.setLevel(logging.DEBUG)


            # Instantiate as many drivers as required inputs to the model
            self.input_drivers = {}
            self.output_monitors = {}

            for node in graph.nodes_in:
                for arg in node.meta["mase"]["common"]["args"].keys():
                    print ("arg in _emit_cocotb:", arg)
                    if "data_in" not in arg:
                        continue
                    print ("getattr(dut, arg)",getattr(dut, arg))
                    print ("getattr(dut, arg).value",getattr(dut, arg).value)
                    self.input_drivers[arg] = StreamDriver(
                        dut.clk,
                        getattr(dut, arg),
                        getattr(dut, f"{arg}_valid"),
                        getattr(dut, f"{arg}_ready"),

                    )
                    self.input_drivers[arg].log.setLevel(logging.DEBUG)

            # Instantiate as many monitors as required outputs
            for node in graph.nodes_out:
                for result in node.meta["mase"]["common"]["results"].keys():
                    if "data_out" not in result:
                        continue
                    self.output_monitors[result] = StreamMonitor(
                        dut.clk,
                        getattr(dut, result),
                        getattr(dut, f"{result}_valid"),
                        getattr(dut, f"{result}_ready"),
                        check=False,
                    )
                    self.output_monitors[result].log.setLevel(logging.DEBUG)

            self.model = graph.model

            # To do: precision per input argument
            self.input_precision = graph.meta["mase"]["common"]["args"]["data_in_0"][
                "precision"
            ]

        def generate_inputs(self, batches):
            """
            Generate inputs for the model by sampling a random tensor
            for each input argument, according to its shape

            :param batches: number of batches to generate for each argument
            :type batches: int
            :return: a dictionary of input arguments and their corresponding tensors
            :rtype: Dict
            """
            print ('nihap in generate_inputs')
            # ! TO DO: iterate through graph.args instead to generalize
            inputs = {}
            for node in graph.nodes_in:
                for arg, arg_info in node.meta["mase"]["common"]["args"].items():
                    # Batch dimension always set to 1 in metadata
                    if "data_in" not in arg:
                        continue
                    # print(f"Generating data for node {node}, arg {arg}: {arg_info}")
                # e.g. [10, 3, 224, 224]: arg_info["shape"][1:] means everything
                # [batches] + arg_info["shape"][1:] means you take the single-element list [batches]
                # (for example, [32]) and then append [3, 224, 224] to get [32, 3, 224, 224].
                    inputs[f"{arg}"] = torch.rand(([batches] + arg_info["shape"][1:]))
                    # print ("inputs in generate_inputs:", inputs)
                    # print ("end")
            return inputs

        # def model(self, inputs):
        #     print ('nihap form model??')
        #     if self.SIGNED:
        #         inputs = [[sign_extend(x, self.DATA_WIDTH) for x in l] for l in inputs]

        #     exp_out = []
        #     for l in inputs:
        #         if self.MAX1_MIN0:
        #             exp_out.append(max(l))
        #         else:
        #             exp_out.append(min(l))

        #     if self.SIGNED:
        #         exp_out = [signed_to_unsigned(x, self.DATA_WIDTH) for x in exp_out]

        #     return exp_out

        def load_drivers(self, in_tensors):
            print ("load_drivers in emit_tb.py")
            # for arg, arg_batches in in_tensors.items():
            #     print ("arg in load_drivers in emit_tb: ",arg)
            #     print ("arg_batch in load_drivers in emit_tb: ",arg_batches)

            for arg, arg_batches in in_tensors.items():

                for key, value in in_tensors.items():
                   if key == 'data_in_0':
                       tensor_dim = value.dim()
                       print ("tensor_dim:", tensor_dim)
                in_tensor_size= in_tensors

                # Build the parallelism list dynamically.
                # If tensor_dim = n, this gives you indices [0, 1, ..., n-2],
                 #    i.e., n-1 parameters total.
                parallelism_list = [
                self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_{i}")
                for i in range(tensor_dim)
                ]
                # parallelism_list= [16,1]
                # reverse the order
                parallelism_list_into_fixed_process= parallelism_list[::-1]

                print ("parallelism_list:",parallelism_list_into_fixed_process )
                # Quantize input tensor according to precision
                if len(self.input_precision) > 1:
                    from mase_cocotb.utils import fixed_preprocess_tensor

                    in_data_blocks = fixed_preprocess_tensor(
                        tensor=arg_batches,
                        q_config={
                            "width": self.get_parameter(f"{_cap(arg)}_PRECISION_0"),
                            "frac_width": self.get_parameter(
                                f"{_cap(arg)}_PRECISION_1"
                            ),
                        },
                        parallelism=parallelism_list_into_fixed_process
                        # parallelism=[
                        #     self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_1"),
                        #     self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_0"),
                        #     # self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_2"),
                        # ],
                    )


                else:
                    # TO DO: convert to integer equivalent of floating point representation
                    pass

                # Append all input blocks to input driver
                # ! TO DO: generalize I help you here::::
                block_size = 1
                for i in range(tensor_dim):
                    block_size *= self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_{i}")
                # block_size = self.get_parameter(
                #     "DATA_IN_0_PARALLELISM_DIM_0"
                # ) * self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1")
                print ("block_size:",block_size)
                print ("in_data_blocks:", in_data_blocks)
                for block in in_data_blocks:
                    if len(block) < block_size:
                        block = block + [0] * (block_size - len(block))
                        # This is where it creates 0s if the data sends in is not as big as block_size
                    print ("block sending in:", block)
                    self.input_drivers[arg].append(block)
                    # print ("self.input_drivers in load_drivers:",self.input_drivers)

        def load_monitors(self, expectation):
            from mase_cocotb.utils import fixed_preprocess_tensor

            print ("expectations:", expectation)
            exp_shape=expectation.dim()
            print ("shape of expectations:", exp_shape)
            parallelism_list = [
                self.get_parameter(f"DATA_OUT_0_PARALLELISM_DIM_{i}")
                for i in range(exp_shape)
                ]
                # parallelism_list= [16,1]
                # reverse the order
            parallelism_list_into_fixed_process= parallelism_list[::-1]
            # Process the expectation tensor
            output_blocks = fixed_preprocess_tensor(
                tensor=expectation,
                q_config={
                    "width": self.get_parameter(f"DATA_OUT_0_PRECISION_0"),
                    "frac_width": self.get_parameter(f"DATA_OUT_0_PRECISION_1"),
                },
                parallelism=parallelism_list_into_fixed_process
                # parallelism=[
                #     self.get_parameter(f"DATA_OUT_0_PARALLELISM_DIM_1"),
                #     self.get_parameter(f"DATA_OUT_0_PARALLELISM_DIM_0"),
                # ],
            )
            print ("output_blocks in load_monitors:", output_blocks)
            # Set expectation for each monitor
            for block in output_blocks:
                # ! TO DO: generalize to multi-output models I help you here:
                if len(block) < self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"):
                    block = block + [0] * (
                        self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0") - len(block)
                    )
                self.output_monitors["data_out_0"].expect(block)

                print ("block in load_monitors in emit_tb.py:", block)

            # Drive the in-flight flag for each monitor
            self.output_monitors["data_out_0"].in_flight = True

    # Serialize testbench object to be instantiated within test by cocotb runner
    cls_obj = MaseGraphTB
    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "tb_obj.dill", "wb") as file:
        dill.dump(cls_obj, file)
    with open(tb_path / "__init__.py", "w") as file:
        file.write("from .test import test")


def emit_cocotb_transform_pass(graph, pass_args={}):
    """
    Emit test bench and related files for simulation

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)

    - pass_args
        - project_dir -> str : the directory of the project
        - trace -> bool : trace waves in the simulation
    """
    logger.info("Emitting testbench...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )

    init_project(project_dir)

    _emit_cocotb_test(graph, pass_args=pass_args)
    _emit_cocotb_tb(graph)

    return graph, None
