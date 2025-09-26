#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

namespace flash {


struct SingleTileScheduler {

public:

    // host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_blocks_n, num_batch;
        int* const tile_count_semaphore = nullptr;
    };

    struct Params {};

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_blocks_n), uint32_t(args.num_batch)};
    }

    struct WorkTileInfo {
        int M_idx = 0;
        int N_idx = 0;
        int B_idx = 0;
        bool is_valid_tile = false;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return is_valid_tile;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, N_idx, B_idx};
        }

    }

    CUTLASS_DEVICE
    SingleTileScheduler(int* tile_count_smem_) {}

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    CUTLASS_DEVICE
    void
    broadcast_next_work(WorkTileInfo& current_work) const {}

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }

};


} // namespace flash