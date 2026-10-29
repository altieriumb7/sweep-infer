; ==============================================================================
; SynapticFold: Native AVX-2 Assembly Kernel for Cortical Minicolumn Activation
; Target: x86_64 Windows (Microsoft x64 ABI)
; Assembled with NASM 3.02+
;
; Computes for each cortical column c in [0, num_columns-1]:
;   potential_c = sum_{j=0}^{D-1}(x[j] * W[c*D + j]) + b_resting[c]
;   out_potentials[c] = (potential_c > v_threshold) ? potential_c : 0.0f
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

synaptic_column_avx2_forward:
    ; Prologue
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Load parameters from stack
    ; [rbp+16] = return address, [rbp+24] = shadow RCX, ...
    ; Shadow space is 32 bytes (4 quadwords):
    ; [rbp + 16 + 32] = 5th arg (D)
    ; [rbp + 16 + 40] = 6th arg (num_columns)
    ; [rbp + 16 + 48] = 7th arg (v_threshold)
    mov r12d, dword [rbp + 48]     ; D
    mov r13d, dword [rbp + 56]     ; num_columns
    vmovss xmm6, dword [rbp + 64]  ; v_threshold
    vbroadcastss ymm6, xmm6        ; ymm6 = [v_thresh, v_thresh, ...]

    ; Zero registers
    vxorps ymm7, ymm7, ymm7        ; ymm7 = [0.0f, ...]

    xor r10d, r10d                 ; col_idx = 0 (R10)
    mov r14, RDX                   ; W_ptr = W_centroids (R14)

.column_loop:
    cmp r10d, r13d
    jge .done

    ; Accumulator for column dot-product
    vxorps ymm0, ymm0, ymm0        ; acc0 = 0.0
    vxorps ymm1, ymm1, ymm1        ; acc1 = 0.0

    xor eax, eax                   ; j = 0

.dim_loop:
    mov edx, r12d
    sub edx, eax                   ; remaining elements
    cmp edx, 16
    jl .dim_loop_8

    ; Unrolled 16 floats (2 x 256-bit AVX-2 vectors = 64 bytes)
    vmovups ymm2, [RCX + rax*4]         ; load x[j..j+7]
    vmovups ymm3, [RCX + rax*4 + 32]    ; load x[j+8..j+15]
    vfmadd231ps ymm0, ymm2, [r14 + rax*4]      ; acc0 += x[0..7] * W[0..7]
    vfmadd231ps ymm1, ymm3, [r14 + rax*4 + 32] ; acc1 += x[8..15] * W[8..15]

    add eax, 16
    jmp .dim_loop

.dim_loop_8:
    cmp edx, 8
    jl .dim_loop_scalar

    vmovups ymm2, [RCX + rax*4]
    vfmadd231ps ymm0, ymm2, [r14 + rax*4]
    add eax, 8
    jmp .dim_loop

.dim_loop_scalar:
    ; Horizontal reduction of ymm0 and ymm1
    vaddps ymm0, ymm0, ymm1

    ; Extract 128-bit high half
    vextractf128 xmm2, ymm0, 1
    vaddps xmm0, xmm0, xmm2            ; xmm0 = [a+e, b+f, c+g, d+h]
    vhaddps xmm0, xmm0, xmm0           ; [ab+ef, cd+gh, ...]
    vhaddps xmm0, xmm0, xmm0           ; sum in xmm0[0]

    ; Any remaining scalar elements (0 to 7)
.scalar_tail:
    cmp eax, r12d
    jge .add_bias

    vmovss xmm2, [RCX + rax*4]
    vmulss xmm2, xmm2, [r14 + rax*4]
    vaddss xmm0, xmm0, xmm2
    inc eax
    jmp .scalar_tail

.add_bias:
    ; Add homeostatic resting potential bias b_resting[c]
    vaddss xmm0, xmm0, [R8 + r10*4]

    ; Apply cortical membrane threshold: if potential < v_threshold -> 0.0f
    vcomiss xmm0, xmm6
    jae .store_potential
    vxorps xmm0, xmm0, xmm0            ; below threshold: clamp to 0.0f

.store_potential:
    vmovss [R9 + r10*4], xmm0          ; out_potentials[c] = potential

    ; Advance W_ptr by D * 4 bytes
    mov eax, r12d
    shl rax, 2
    add r14, rax

    inc r10d                           ; next column
    jmp .column_loop

.done:
    ; Clean AVX state to prevent transition penalties
    vzeroupper

    ; Epilogue
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
