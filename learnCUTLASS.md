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