       device::Arguments
              │
              │
              ▼
┌────────────────────────────┐
│ problem_size               │
│ ref_A = pointer + layout   │
│ ref_B = pointer + layout   │
│ ref_C = pointer + layout   │
│ ref_D = pointer + layout   │
│ epilogue(alpha,beta...)    │
│ split_k                    │
└────────────────────────────┘
              │
              │ initialize
              ▼
┌────────────────────────────┐
│     GemmKernel::Params     │
│                            │
│ params_A ← layout/stride A │
│ params_B ← layout/stride B │
│ params_C ← layout/stride C │
│ params_D ← layout/stride D │
│                            │
│ ptr_A                      │
│ ptr_B                      │
│ ptr_C                      │
│ ptr_D                      │
│                            │
│ output_op ← epilogue       │
│ grid/tile/splitK info      │
└────────────────────────────┘
              │
              ▼
         CUDA kernel


Gemm<...>
  │
  │ 编译期
  ↓
确定 kernel 类型
dtype/layout/tile/arch...

Arguments
  │
  │ 运行时用户输入
  ↓
M/N/K
A/B/C/D
epilogue
split-K

  ↓ initialize()

Params
  ↓
iterator params
grid tiled shape
gemm_k_size
workspace/semaphore
...

  ↓ run()

grid
block
smem
stream

  ↓

Kernel<GemmKernel>
<<<...>>>(params_)

现在不需要把 block/grid/workspace 的所有计算细节都抠到最底。你先掌握到“知道它们分别由什么决定、服务什么目的”就够了，然后直接往 cutlass::Kernel<GemmKernel> 深入，收益更高。

你现在对这三个东西至少要有下面这个清晰度：

grid：由 problem_size、ThreadblockShape、split-K、ThreadblockSwizzle 决定。最简单时就是 ceil(M/BM) × ceil(N/BN)，再考虑 split-K。
block：由 GemmKernel::kThreadCount 决定，本质上来自这个 GEMM kernel 里面配置了多少 warp / thread。
workspace：不是每个 CTA 的 shared memory，而是整个 GEMM 调用额外需要的 global-memory 辅助空间，经典场景就是 split-K semaphore。
smem_size：这个反而要区分清楚，它才是每个 CTA 的 shared memory，来自 sizeof(GemmKernel::SharedStorage)。

其中 grid/block 后面进入 kernel 后我们还会重新碰到，所以现在不必停下来死磕。

CUTLASS_DEVICE
void operator()

我们接下来要看的就是：
GemmKernel::operator()
这是第一个真正开始出现 GEMM 执行逻辑的地方。

进去以后只抓 5 件事

CUTLASS 确实繁琐，所以我们继续限制阅读范围，不要每个类型都点。

在 GemmKernel::operator() 里，我们只看：

当前 block 负责哪个 tile

blockIdx
   ↓
tile_offset

这个 block 负责哪一段 K

gemm_k_size
split-K

A/B iterator 怎么创建

IteratorA
IteratorB

MMA mainloop 在哪调用

Mma mma(...)
mma(...)

Accumulator 最后怎么交给 Epilogue

accumulator
   ↓
epilogue
   ↓
D

一旦找到这五个，你会得到真正的数据流：

blockIdx
   ↓
确定 C tile
   ↓
定位 A tile / B tile
   ↓
构造 IteratorA / IteratorB
   ↓
MMA mainloop
   ↓
Accumulator
   ↓
Epilogue
   ↓
D

这就开始和你自己写 CUDA GEMM 的代码高度对应了。

你现在这里先记死一句就行：

threadblock_tile_offset.m/n
= 当前 CTA 负责哪个 C tile


threadblock_tile_offset.k
= 当前 CTA 属于哪个 K partition（普通 GEMM 通常就是 0）


真正遍历 K 的循环
= 后面的 gemm_k_iterations / Mma mainloop

把这个搞清楚之后，再往下看代码时，看到 threadblock_tile_offset.k() 就不会误以为它是那个 for(k...) 的 k 了。


GemmKernel::operator()
{
    1. 当前 CTA 对应哪个 GEMM tile？

    2. A/B 从哪里开始读？

    3. 构造 A/B iterator

    4. 创建 MMA mainloop 对象

    5. 沿 K 维循环：
         Global -> Shared
         Shared -> Register
         MMA accumulate

    6. Epilogue:
         accumulator -> D
}

你现在如果能不看源码说出下面这一段，就够了：

GemmKernel::operator() 首先通过 ThreadblockSwizzle 将 CUDA 的 block 坐标映射成 GEMM 的逻辑 tile 坐标，然后根据 tile 坐标计算 A、B 当前 threadblock 的起始位置，并构造 IteratorA/IteratorB 管理 global-memory tile 的访问。之后构造 Mma mainloop 对象，Mma 根据 thread/warp/lane 的身份把 CTA 内部工作划分到各线程。调用 mma(...) 后沿 K 维进行多次 tile 迭代，把 A/B 数据加载到 shared memory / registers，并执行 MMA 累加，最终生成 accumulator fragment，之后交给 epilogue 写回输出。

这个程度已经非常好了。

甚至面试讲 CUTLASS 架构，这段都已经挺像样了。

Mma::operator()
1. gemm_k_iterations 是多少？

2. 一轮 K iteration 搬多少 A/B？

3. global → shared 在哪？

4. shared → register → MMA 在哪？

include/cutlass/gemm/threadblock/mma_pipelined.h

IteratorA/B
    ↓
Global Memory iterator

FragmentA/B
    ↓
从 global load 到线程寄存器里的临时数据

SmemIteratorA/B
    ↓
负责把数据写进 Shared Memory

warp_tile_iterator_A/B
    ↓
从 Shared Memory 给 warp 取数据

Operator / warp_mma
    ↓
真正的 warp-level MMA

FragmentC
    ↓
累加器

// prologue
load_global(A_tile0, reg_A);
load_global(B_tile0, reg_B);

store_shared(reg_A, As[0]);
store_shared(reg_B, Bs[0]);

__syncthreads();

accum = 0;

// load first warp fragment
load_shared(As[0], warp_A[0]);
load_shared(Bs[0], warp_B[0]);

for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {

    for (int warp_k = 0; warp_k < warp_k_iters; ++warp_k) {

        // prepare next warp fragment
        load_shared(
            As[current_stage],
            warp_A[(warp_k + 1) % 2]);

        load_shared(
            Bs[current_stage],
            warp_B[(warp_k + 1) % 2]);

        // prefetch next CTA K tile
        if (warp_k == 0) {
            load_global(next_A, reg_A);
            load_global(next_B, reg_B);
        }

        // near the end, put prefetched tile
        // into alternate shared buffer
        if (warp_k == warp_k_iters - 1) {

            store_shared(reg_A, As[next_stage]);
            store_shared(reg_B, Bs[next_stage]);

            __syncthreads();

            swap_stage();
        }

        // actual GEMM
        warp_mma(
            accum,
            warp_A[warp_k % 2],
            warp_B[warp_k % 2],
            accum);
    }
}