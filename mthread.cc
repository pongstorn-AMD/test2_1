// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// ONLY TO TEST MULTTHREAD SUPPORT OF LIBRRAY AND ENGINE
// g++ mthread.cc -lpthread -o mthread

#include <cassert>
#include <iostream>
#include <thread>
#include <vector>


#if ENABLE_GEM5==1
#pragma message("Compiling with gem5 instructions")
#include <gem5/m5ops.h>
#include "m5_mmap.h"
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE==1
#pragma message("Compiling with Pickle device")
#include "pickle_graph.h"
std::unique_ptr<PickleDeviceManager> pdev(new PickleDeviceManager());
uint64_t* UCPage = NULL;
#endif // ENABLE_PICKLEDEVICE


using namespace std;

// A callable object
class thread_obj {
private:
    int _id;
public:
    thread_obj(int id):_id(id) {}
    void operator()(int x)
    {
        for (int i = 0; i < x; i++) {
	    #if ENABLE_PICKLEDEVICE==1
                //cerebellum_manager->sendWork((uint64_t)(i),_id);
		*UCPage = (uint64_t)(_id);
                cout << "DevThread " << _id << " " << i << "\n";
            #else  		
                cout << "Thread " << _id << " " << i << "\n";
            #endif
        }
    }
};

// Driver code
int main(int argc, char** argv)
{
    int num = 100;
    int nt  = 8;
    int dly  = 8; // engine cycles
    if (argc >= 2)
      nt = atoi(argv[1]);
    if (argc >= 3)
      num = atoi(argv[2]);
    if (argc >= 4)
      dly = atoi(argv[3]);

    cout << "num Threads " << nt << " num iteration " << num << " delay in engine for " << dly << endl;

    #if ENABLE_GEM5==1
    map_m5_mem();
    //m5_work_begin_addr(0, 0); // 1st stat dump
    #endif // ENABLE_GEM5

    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5
    std::cout << "mthread checkpointing" << std::endl;

    #if ENABLE_PICKLEDEVICE==1
    // why do I need this?
    //PerfPage = (uint64_t*) pdev->getPerfPagePtr();
    //std::cout << "PerfPage: 0x" << std::hex << (uint64_t)PerfPage << std::dec << std::endl;
    //assert(PerfPage != nullptr);
    #endif

// don't need it here    
//    #if ENABLE_GEM5==1
//    m5_exit_addr(0);
//    #endif // ENABLE_GEM5
//    std::cout << "mthread extra exit" << std::endl;

    uint64_t use_pdev = 0;
    #if ENABLE_PICKLEDEVICE==1
    PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
    use_pdev = specs.availability;
    //prefetch_distance = specs.prefetch_distance;
    //prefetch_mode = specs.prefetch_mode;
    //bulk_mode_chunk_size = specs.bulk_mode_chunk_size;
    #endif

    #if ENABLE_PICKLEDEVICE==1
    assert(use_pdev != 0);	    
    UCPage = (uint64_t*) pdev->getUCPagePtr(0);
    std::cout << "UCPage: 0x" << std::hex << (uint64_t)UCPage << std::dec << std::endl;
    assert(UCPage != nullptr);
    #endif // ENABLE_PICKLEDEVICE


    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5
    std::cout << "mthread ROI Start" << std::endl;


    std::vector<std::thread> ThreadVector;
    for (int t = 0; t < nt; ++t) {
      thread th(thread_obj(t), num);
      ThreadVector.push_back(std::move(th));  //<=== move (after, th doesn't hold it anymore
    }

    for(auto& th : ThreadVector){              //<=== range-based for uses & reference
      th.join();
    }


    std::cout << "mthread ROI End" << std::endl;
    #if ENABLE_GEM5==1
    m5_exit_addr(0);

    #endif // ENABLE_GEM5
    std::cout << "mthread ROI End" << std::endl;
    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5
    std::cout << "mthread ROI End" << std::endl;
    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5


    cout << "all threads finished\n";

    #if ENABLE_GEM5==1
    //m5_work_end_addr(0, 0);
    //unmap_m5_mem();
    #endif // ENABLE_GEM5

    return 0;
}
