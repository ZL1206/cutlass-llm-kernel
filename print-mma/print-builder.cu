#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/detail/collective.hpp"

using namespace cute;



int main() {


    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using CollectiveBuilder = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >;



    using ExtractedElementA = cutlass::gemm::collective::detail::deduce_mixed_width_dtype_t<0, cutlass::half_t>;
    print(ExtractedElementA{}); print("\n");
    //print_type<ExtractedElementA>();
    std::cout << typeid(ExtractedElementA).name() << std::endl;  // Fallback



    std::cout << CollectiveBuilder::name << std::endl; 


    using Builder = typename CollectiveBuilder::CollectiveOp;

    using CollectiveMainloop = typename Builder::CollectiveOp;

    std::cout << Builder::name << std::endl;

    cute::GMMA::Major GmmaMajorA = Builder::GmmaMajorA;
    cute::GMMA::Major GmmaMajorB = Builder::GmmaMajorB;
    
    print("GmmaMajorA: "); print(static_cast<int>(GmmaMajorA)); print("\n");
    print("GmmaMajorB: "); print(static_cast<int>(GmmaMajorB)); print("\n");

    print("AtomLayoutMNK: "); print(typename Builder::AtomLayoutMNK{}); print("\n");

    print("TiledMma: "); print(typename Builder::TiledMma{}); print("\n");

    print("SmemLayoutAtomA: "); print(typename Builder::SmemLayoutAtomA{}); print("\n");
    print("SmemLayoutAtomB: "); print(typename Builder::SmemLayoutAtomB{}); print("\n");
    print_layout(typename Builder::SmemLayoutAtomA{});

    constexpr int mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
    printf("mainloop_pipeline_bytes: %d\n", mainloop_pipeline_bytes);

    int PipelineStages = Builder::PipelineStages;
    printf("PipelineStages is %d\n", PipelineStages);

    std::cout << CollectiveMainloop::name << std::endl;


    // epilogue
    using CollectiveEpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >;

    std::cout << CollectiveEpilogueBuilder::name << std::endl;

    using EpilogueBuilder = typename CollectiveEpilogueBuilder::CollectiveOp;
    std::cout << EpilogueBuilder::name << std::endl;

    using CollectiveEpilogue = typename EpilogueBuilder::CollectiveOp;

    std::cout << CollectiveEpilogue::name << std::endl;

    // gemm kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue
    >;
    std::cout << GemmKernel::name << std::endl;


    return 0;
}