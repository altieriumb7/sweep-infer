; ==============================================================================
; SynapticFold: Optimized AVX-2 Assembly Kernel for Cortical Minicolumns
; Target: AMD Zen 3 (Ryzen 7 7730U) / x86_64 Windows (Microsoft x64 ABI)
; Optimization: 4-Way Column Blocking + 8 Accumulator Unrolling + L1 Prefetch
;
; Evaluates 4 cortical columns simultaneously per inner loop:
;   - Vector x is loaded once into YMM register and reused across 4 columns.
;   - 8 independent FMA accumulators hide the 4-cycle latency of Zen 3 pipes.
;   - Cuts memory loads of x by 75%, maximizing L1D cache throughput.
;
; Parameters (Windows x64 ABI):
;   RCX: const float* x               (input vector of size D)
;   RDX: const float* W_centroids     (flattened matrix: num_columns x D)
;   R8:  const float* b_resting       (column resting biases: num_columns)
;   R9:  float*       out_potentials  (output activation potentials: num_columns)
;   [RSP+40]: int     D               (feature dimension, multiple of 8)
;   [RSP+48]: int     num_columns     (number of cortical columns)
;   [RSP+56]: float   v_threshold     (membrane activation threshold)
; ==============================================================================

section .text
default rel
global synaptic_column_avx2_forward
global synaptic_column_avx2_forward_opt

synaptic_column_avx2_forward:
synaptic_column_avx2_forward_opt:
    ; Prologue
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Load parameters from stack
    mov r12d, dword [rbp + 48]     ; D (dimension)
    mov r13d, dword [rbp + 56]     ; num_columns
    vmovss xmm6, dword [rbp + 64]  ; v_threshold
    vbroadcastss ymm6, xmm6        ; ymm6 = [v_thresh, ...]

    vxorps ymm7, ymm7, ymm7        ; ymm7 = [0.0f, ...]

    xor r10d, r10d                 ; col_idx = 0 (R10)
    mov r14, RDX                   ; W_ptr = W_centroids (R14)

.col4_loop:
    ; Check if at least 4 columns remaining
    mov eax, r13d
    sub eax, r10d
    cmp eax, 4
    jl .scalar_col_loop

    ; Setup pointers to 4 columns:
    ; c0 = r14
    ; c1 = r14 + D*4
    ; c2 = r14 + 2*D*4
    ; c3 = r14 + 3*D*4
    mov r11, r14                   ; c0_ptr
    mov eax, r12d
    shl rax, 2                     ; D_bytes = D * 4
    lea rbx, [r11 + rax]           ; c1_ptr
    lea r15, [rbx + rax]           ; c2_ptr
    lea rdx, [r15 + rax]           ; c3_ptr

    ; 8 Accumulators for 4 columns:
    ; col0: ymm0, ymm1
    ; col1: ymm2, ymm3
    ; col2: ymm4, ymm8
    ; col3: ymm9, ymm10
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    vxorps ymm8, ymm8, ymm8
    vxorps ymm9, ymm9, ymm9
    vxorps ymm10, ymm10, ymm10

    xor eax, eax                   ; j = 0

.dim4_loop:
    mov esi, r12d
    sub esi, eax
    cmp esi, 16
    jl .dim4_loop_8

    ; Prefetch weights ahead for next iteration
    prefetcht0 [r11 + rax*4 + 128]
    prefetcht0 [rbx + rax*4 + 128]

    ; Load x[j..j+7] into ymm11 once, reuse across all 4 columns!
    vmovups ymm11, [RCX + rax*4]
    vfmadd231ps ymm0, ymm11, [r11 + rax*4]
    vfmadd231ps ymm2, ymm11, [rbx + rax*4]
    vfmadd231ps ymm4, ymm11, [r15 + rax*4]
    vfmadd231ps ymm9, ymm11, [rdx + rax*4]

    ; Load x[j+8..j+15] into ymm12 once, reuse across all 4 columns!
    vmovups ymm12, [RCX + rax*4 + 32]
    vfmadd231ps ymm1, ymm12, [r11 + rax*4 + 32]
    vfmadd231ps ymm3, ymm12, [rbx + rax*4 + 32]
    vfmadd231ps ymm8, ymm12, [r15 + rax*4 + 32]
    vfmadd231ps ymm10, ymm12, [rdx + rax*4 + 32]

    add eax, 16
    jmp .dim4_loop

.dim4_loop_8:
    cmp esi, 8
    jl .dim4_reduce

    vmovups ymm11, [RCX + rax*4]
    vfmadd231ps ymm0, ymm11, [r11 + rax*4]
    vfmadd231ps ymm2, ymm11, [rbx + rax*4]
    vfmadd231ps ymm4, ymm11, [r15 + rax*4]
    vfmadd231ps ymm9, ymm11, [rdx + rax*4]

    add eax, 8

.dim4_reduce:
    ; Sum high and low halves for each column
    vaddps ymm0, ymm0, ymm1        ; col0 total ymm
    vaddps ymm2, ymm2, ymm3        ; col1 total ymm
    vaddps ymm4, ymm4, ymm8        ; col2 total ymm
    vaddps ymm9, ymm9, ymm10       ; col3 total ymm

    ; Perform reduction of col0:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0       ; sum col0 in xmm0[0]

    ; Perform reduction of col1:
    vextractf128 xmm3, ymm2, 1
    vaddps xmm2, xmm2, xmm3
    vhaddps xmm2, xmm2, xmm2
    vhaddps xmm2, xmm2, xmm2       ; sum col1 in xmm2[0]

    ; Perform reduction of col2:
    vextractf128 xmm5, ymm4, 1
    vaddps xmm4, xmm4, xmm5
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4       ; sum col2 in xmm4[0]

    ; Perform reduction of col3:
    vextractf128 xmm11, ymm9, 1
    vaddps xmm9, xmm9, xmm11
    vhaddps xmm9, xmm9, xmm9
    vhaddps xmm9, xmm9, xmm9       ; sum col3 in xmm9[0]

    ; Add resting biases for the 4 columns
    vaddss xmm0, xmm0, [R8 + r10*4]
    vaddss xmm2, xmm2, [R8 + r10*4 + 4]
    vaddss xmm4, xmm4, [R8 + r10*4 + 8]
    vaddss xmm9, xmm9, [R8 + r10*4 + 12]

    ; Threshold gating for all 4
    vcomiss xmm0, xmm6
    jae .store0
    vxorps xmm0, xmm0, xmm0
.store0:
    vmovss [R9 + r10*4], xmm0

    vcomiss xmm2, xmm6
    jae .store1
    vxorps xmm2, xmm2, xmm2
.store1:
    vmovss [R9 + r10*4 + 4], xmm2

    vcomiss xmm4, xmm6
    jae .store2
    vxorps xmm4, xmm4, xmm4
.store2:
    vmovss [R9 + r10*4 + 8], xmm4

    vcomiss xmm9, xmm6
    jae .store3
    vxorps xmm9, xmm9, xmm9
.store3:
    vmovss [R9 + r10*4 + 12], xmm9

    ; Advance W_ptr by 4 * D * 4 bytes
    mov eax, r12d
    shl rax, 4                     ; 4 * D * 4 = 16 * D
    add r14, rax

    add r10d, 4
    jmp .col4_loop

; Remainder scalar column loop for any trailing columns
.scalar_col_loop:
    cmp r10d, r13d
    jge .done

    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    xor eax, eax

.dim_single_loop:
    mov edx, r12d
    sub edx, eax
    cmp edx, 16
    jl .dim_single_8

    vmovups ymm2, [RCX + rax*4]
    vmovups ymm3, [RCX + rax*4 + 32]
    vfmadd231ps ymm0, ymm2, [r14 + rax*4]
    vfmadd231ps ymm1, ymm3, [r14 + rax*4 + 32]
    add eax, 16
    jmp .dim_single_loop

.dim_single_8:
    cmp edx, 8
    jl .dim_single_scalar
    vmovups ymm2, [RCX + rax*4]
    vfmadd231ps ymm0, ymm2, [r14 + rax*4]
    add eax, 8
    jmp .dim_single_loop

.dim_single_scalar:
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm2, ymm0, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

.scalar_tail_single:
    cmp eax, r12d
    jge .add_bias_single
    vmovss xmm2, [RCX + rax*4]
    vmulss xmm2, xmm2, [r14 + rax*4]
    vaddss xmm0, xmm0, xmm2
    inc eax
    jmp .scalar_tail_single

.add_bias_single:
    vaddss xmm0, xmm0, [R8 + r10*4]
    vcomiss xmm0, xmm6
    jae .store_single
    vxorps xmm0, xmm0, xmm0
.store_single:
    vmovss [R9 + r10*4], xmm0
    mov eax, r12d
    shl rax, 2
    add r14, rax
    inc r10d
    jmp .scalar_col_loop

.done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
