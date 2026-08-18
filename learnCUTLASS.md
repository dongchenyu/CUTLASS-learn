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