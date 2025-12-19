import argparse
from pathlib import Path
import time

import m5

from gem5.utils.requires import requires
from gem5.utils.override import overrides
from gem5.components.boards.arm_board import ArmBoard
from gem5.components.memory.dram_interfaces.ddr4 import DDR4_2400_8x8
from gem5.components.memory.dram_interfaces.ddr5 import DDR5_8400_4x8
from gem5.components.memory.memory import ChanneledMemory
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.coherence_protocol import CoherenceProtocol
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent
from gem5.resources.workload import Workload
from gem5.resources.resource import (
    Resource,
    CustomResource,
    CustomDiskImageResource,
    obtain_resource,
)
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)

from MeshCache.MeshCache import MeshCache
from MeshCache.MeshCacheWithPickleDevice import MeshCacheWithPickleDevice
from MeshCache.components.PrebuiltMesh import PrebuiltMesh

from m5.objects import (
    PickleDevice,
    TrafficSnooper,
    AddrRange,
    ArmMMU,
    PickleDeviceRequestManager,
    PicklePrefetcher,
    TAGE_SC_L_64KB,
)

from m5.objects import (
    ArmDecoder,
    ArmDefaultRelease,
    ArmISA,
    VExpress_GEM5_V1,
    VExpress_GEM5_Foundation,
)

mesh_descriptor = PrebuiltMesh.getMesh8("Mesh8")

parser = argparse.ArgumentParser()
parser.add_argument("--application", type=str, required=True, choices={"bfs", "pr", "tc", "cc", "spmv"})
parser.add_argument("--graph_name", type=str, required=True)
parser.add_argument("--single_threaded", type=str, required=True, choices={"True", "False"})
# parser.add_argument("--enable_pdev", type=str, required=True, choices=["True", "False"])
args = parser.parse_args()

application = args.application
graph_name = args.graph_name
single_threaded = args.single_threaded == "True"
# from _m5.core import setOutputDir
# setOutputDir(f"/workdir/ARTIFACTS/results/bfs-pickle-{graph_name}-distance-32")

num_cores = mesh_descriptor.get_num_core_tiles()

fast_forward_cpu_type = CPUTypes.KVM

special_memory_requirement = {
    ("spmv", "nlpkkt200"): "8GiB",
    ("spmv", "nlpkkt240"): "16GiB",
}
def choose_memory_size(application, graph_name):
    if (application, graph_name) in special_memory_requirement:
        return special_memory_requirement[(application, graph_name)]
    return "4GiB"

mesh_cache = MeshCacheWithPickleDevice(
    l1i_size="32KiB",
    l1i_assoc=8,
    l1d_size="48KiB",
    l1d_assoc=12,
    l2_size="1MiB",
    l2_assoc=16,
    l3_size="32MiB",
    l3_assoc=16,
    device_cache_size="32KiB",
    device_cache_assoc=8,
    num_core_complexes=1,
    is_fullsystem=True,
    mesh_descriptor=mesh_descriptor,
    data_prefetcher_class=None,
    pdev_num_tbes=16,
)

# Main memory
memory = ChanneledMemory(
    dram_interface_class=DDR5_8400_4x8,
    num_channels=mesh_descriptor.get_num_mem_tiles(),
    interleaving_size=64,
    size=choose_memory_size(application, graph_name),
)

processor = SimpleProcessor(cpu_type=CPUTypes.KVM, isa=ISA.ARM, num_cores=num_cores)

# Here we tell the KVM CPU (the starting CPU) not to use perf.
if fast_forward_cpu_type == CPUTypes.KVM:
    for proc in processor.get_cores():
        proc.core.usePerf = False


class PickleArmBoard(ArmBoard):
    def __init__(self, clk_freq, processor, memory, cache_hierarchy, release, platform):
        super().__init__(
            clk_freq=clk_freq,
            processor=processor,
            memory=memory,
            cache_hierarchy=cache_hierarchy,
            release=release,
            platform=platform,
        )

    @overrides(ArmBoard)
    def get_default_kernel_args(self):
        # The default kernel string is taken from the devices.py file.
        return [
            "console=ttyAMA0",
            "lpj=19988480",
            "norandmaps",
            "root=/dev/vda1",
            "disk_device=/dev/vda1",
            "rw",
            f"mem={self.get_memory().get_size()}",
            "init=/home/ubuntu/gem5-init.sh",
        ]

    @overrides(ArmBoard)
    def _pre_instantiate(self, full_system):
        num_PD_tiles = (
            self.cache_hierarchy.get_mesh_descriptor().get_num_pickle_device_tiles()
        )
        all_cores = [core.core for core in self.processor.get_cores()]
        self.traffic_snoopers = [
            TrafficSnooper(watch_ranges=[AddrRange(0x10110000, 0x10130000)])
            for i in range(num_PD_tiles * len(all_cores))
        ]
        self.pickle_device_mmus = [
            ArmMMU(release_se=ArmDefaultRelease()) for _ in range(num_PD_tiles)
        ]
        self.pickle_device_isas = [ArmISA() for _ in range(num_PD_tiles)]
        self.pickle_device_decoders = [
            ArmDecoder(isa=self.pickle_device_isas[i]) for i in range(num_PD_tiles)
        ]
        self.pickle_device_request_manager = [
            PickleDeviceRequestManager() for i in range(num_PD_tiles)
        ]
        self.pickle_device_prefetchers = [
            PicklePrefetcher(
                software_hint_prefetch_distance=1,
                prefetch_distance_offset_from_software_hint=0,
                num_cores=len(all_cores),
                expected_number_of_prefetch_generators=2,
                concurrent_work_item_capacity=64,
                prefetch_dropping_distance=16,
            )
            for i in range(num_PD_tiles)
        ]
        self.pickle_devices = [
            PickleDevice(
                mmu=self.pickle_device_mmus[i],
                isa=self.pickle_device_isas[i],
                decoder=self.pickle_device_decoders[i],
                device_id=i,
                associated_cores=all_cores[
                    i * len(all_cores) : (i + 1) * len(all_cores)
                ],
                num_cores=len(all_cores),
                request_manager=self.pickle_device_request_manager[i],
                prefetcher=self.pickle_device_prefetchers[i],
                core_to_pickle_latency_in_ticks=250,
                ticks_per_cycle=250,
                uncacheable_forwarders=self.traffic_snoopers[
                    i * len(all_cores) : (i + 1) * len(all_cores)
                ],
            )
            for i in range(num_PD_tiles)
        ]
        self.cache_hierarchy.set_pickle_devices(self.pickle_devices)
        self.cache_hierarchy.set_traffic_uncacheable_forwarders(self.traffic_snoopers)
        super()._pre_instantiate()

    @overrides(ArmBoard)
    def _post_instantiate(self):
        super()._post_instantiate()
        self.cache_hierarchy.post_instantiate()


board = PickleArmBoard(
    clk_freq="4GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=mesh_cache,
    release=ArmDefaultRelease.for_kvm(),
    platform=VExpress_GEM5_V1(),
)

graph_path_map = {
    "amazon": (
        "/home/ubuntu/graphs/amazon.el",
        "undirected",
        "109638",
    ),  # 334863 / 1851736
    "as_skitter": (
        "/home/ubuntu/graphs/as_skitter.el",
        "undirected",
        "1"
    ), # 1696415 / 11095298
    "gplus": (
        "/home/ubuntu/graphs/gplus_newid.el",
        "directed",
        "21508",
    ),  # 105630 / 13648443
    "higgs": ("/home/ubuntu/graphs/higgs.el", "directed", "91265"),  # 360493 / 14120914
    "livejournal": (
        "/home/ubuntu/graphs/livejournal.el",
        "directed",
        "3",
    ),  # 3984967 / 34235868
    "orkut": (
        "/home/ubuntu/graphs/orkut.el",
        "undirected",
        "614125",
    ),  # 3072441 / 234370158
    "pokec": (
        "/home/ubuntu/graphs/pokec.el",
        "directed",
        "326348",
    ),  # 1504295 / 30159128
    "roadNetCA": (
        "/home/ubuntu/graphs/roadNetCA.el",
        "undirected",
        "393999",
    ),  # 1957027 / 5520774
    "twitch": (
        "/home/ubuntu/graphs/twitch.el",
        "undirected",
        "33600",
    ),  # 168114 / 13595114
    "youtube": (
        "/home/ubuntu/graphs/youtube.el",
        "undirected",
        "231414",
    ),  # 1134890 / 5975240
    "web_berkstan": (
        "/home/ubuntu/graphs/web_berkstan.el",
        "directed",
        "1",
    ),  # 459831 / 5310647
    "web_google": (
        "/home/ubuntu/graphs/web_google.el",
        "directed",
        "1",
    ),  # 600493 / 3874195
    "wiki_talk": (
        "/home/ubuntu/graphs/wiki_talk.el",
        "directed",
        "2",
    ),  # 2354316 / 4949282
    "wiki_topcats": (
        "/home/ubuntu/graphs/wiki_topcats.el",
        "directed",
        "358064",
    ),  # 1791489 / 28508141
    "test5": ("/home/ubuntu/graphs/synth_5.el", "undirected", None),
    "test10": ("/home/ubuntu/graphs/synth_10.el", "undirected", None),
}

matrix_path_map = {
    "steam1": "/home/ubuntu/mm/steam1/steam1.csr",
    "nlpkkt200": "/home/ubuntu/mm/nlpkkt200/nlpkkt200.csr",
    "consph": "/home/ubuntu/mm/consph/consph.csr",
    "roadnet": "/home/ubuntu/mm/USA-road-d.USA.csr",
    "Ga41As41H72": "/home/ubuntu/mm/Ga41As41H72/Ga41As41H72.csr",
}

command_prefix = ""
if single_threaded:
    # here we pin the app to core 1 and run on 1 thread
    command_prefix = "export OMP_NUM_THREADS=1; taskset -c 1"

if application in {"bfs", "pr", "tc", "cc"}:
    graph_path, direction, starting_node = graph_path_map[graph_name]
    is_directed_graph = direction == "directed"
    symmetric_flag = "-s" if not is_directed_graph else ""
    starting_node_flag = ""
    if application == "bfs": # we need a starting node in BFS to reliably walk through a large cluster
        if not starting_node:
            starting_node_flag = ""
        else:
            starting_node_flag = f"-r {starting_node}"
    if application == "tc": # tc requires the input graph to be undirected
        if is_directed_graph:
            symmetric_flag = "-s"
            #assert False, f"tc requires the input graph to be undirected"
    #command = f"{command_prefix} /home/ubuntu/gapbs/{application}2.hw.pdev.m5 -n 2 -f {graph_path} {symmetric_flag} {starting_node_flag}"
    num_threads = 4
    num_iterations = 10
    num_dly_cycles = 68
    command = f"/home/ubuntu/gapbs/mthread.hw.pdev.m5 {num_threads} {num_iterations} {num_dly_cycles}"
elif application in {"spmv"}:
    graph_path = matrix_path_map[graph_name]
    command = f"{command_prefix} /home/ubuntu/benchmarks/spmv/spmv.hw.pdev.m5 {graph_path}"
else:
    assert False, f"Unknown application: {application}"

board.set_kernel_disk_workload(
    kernel=CustomResource("/workdir/ARTIFACTS/linux_src/vmlinux"),
    disk_image=CustomDiskImageResource("/workdir/ARTIFACTS/arm64.img.v2"),
    #bootloader=obtain_resource("arm64-bootloader", resource_version="1.0.0"),
    bootloader=CustomResource("/workdir/.cache/gem5/arm64-bootloader"),
    readfile_contents=command,
)


def handle_exit():
    print("exit 1")
    yield True


def handle_work_begin():
    print("Workbegin")
    print("Should not be here")
    assert False
    yield True


def handle_work_end():
    print("Workend")
    print("Should not be here")
    assert False
    yield True


simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.EXIT: handle_exit(),
        ExitEvent.WORKBEGIN: handle_work_begin(),
        ExitEvent.WORKEND: handle_work_end(),
    },
)
# simulator.override_outdir(output_path)

# We maintain the wall clock time.

globalStart = time.time()

print("Running the simulation")

# We start the simulation.
simulator.run()

checkpoint_name = f"{application}-{graph_name}"
if single_threaded:
    checkpoint_name += "-single_threaded"
simulator.save_checkpoint(Path(f"/workdir/ARTIFACTS/checkpoints/{checkpoint_name}"))

print(f"Ran a total of {simulator.get_current_tick() / 1e12} simulated seconds")

print(
    "Total wallclock time: %.2fs, %.2f min"
    % (time.time() - globalStart, (time.time() - globalStart) / 60)
)

print("Exit cause: ", simulator.get_last_exit_event_cause())
