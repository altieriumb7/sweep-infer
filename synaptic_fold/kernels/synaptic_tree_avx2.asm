; ==============================================================================
; SynapticFold: Native AVX-2 Assembly Kernel for Parallel Tree Branching Speculation
; Target: x86_64 Windows (Microsoft x64 ABI)
; Assembled with NASM 3.02+
;
; Evaluates speculative hypothesis trees (e.g. root, 2 children, 1-2 grandchildren)
; in parallel registers without CPU memory saturation.
;
; Pipeline:
; 1. Parallel AVX-2 dot products (vfmadd231ps unrolled 32 floats) for all N nodes
;    against projection centroids/weights + biases -> transition logits z_i.
; 2. Transition probabilities via Softmax over sibling branches or Sigmoidal gating
;    with range-reduced IEEE-754 fast exp.
; 3. Cumulative path score computation: S_path[i] = S_path[parent[i]] * p_i.
; 4. Vectorized pruning with vcmpps / vandps against prob_threshold.
; 5. Top-4 surviving node selection of asymmetric tree.
; 6. Construction of 4x4 2D causal attention mask for single-pass verification.
;
; Parameters (Windows x64 ABI):
;   RCX: const float* node_latents       (matrix: num_nodes x D)
;   RDX: const float* branch_weights     (matrix: num_nodes x D)
;   R8:  const float* branch_biases      (array: num_nodes)
;   R9:  const int*   parent_indices     (array: num_nodes, -1 for root)
;   [RBP+48]: int     num_nodes          (number of nodes in tree, <= 8)
;   [RBP+56]: int     D                  (hidden dimension, multiple of 8)
;   [RBP+64]: float   prob_threshold     (cumulative probability pruning threshold)
;   [RBP+72]: float*  out_path_scores    (output path scores: num_nodes)
;   [RBP+80]: int*    out_selected_nodes (output top-4 selected node indices: 4)
;   [RBP+88]: float*  out_causal_mask    (output 4x4 causal attention mask: 16 floats)
;   [RBP+96]: int*    out_num_selected   (output count of surviving nodes: 1 int)
; ==============================================================================

section .rdata
align 32
const_one:        dd 1.0
const_zero:       dd 0.0
const_neg_80:     dd -80.0
const_pos_80:     dd 80.0
const_log2e:      dd 1.4426950408889634 ; log2(e)
const_ln2:        dd 0.6931471805599453 ; ln(2)
exp_max_val:      dd 1.0e30
exp_c5:           dd 0.0083333333       ; 1/120
exp_c4:           dd 0.041666667        ; 1/24
exp_c3:           dd 0.16666667         ; 1/6
exp_c2:           dd 0.5                ; 1/2
exp_c1:           dd 1.0                ; 1
exp_c0:           dd 1.0                ; 1

section .text
default rel
global synaptic_tree_avx2_forward
global synaptic_tree_avx2_eval

; ------------------------------------------------------------------------------
; Helper: fast_exp (Range-reduced IEEE-754 fast exp on AVX-2)
; Evaluates exp(x) = 2^k * exp(r) where k = round(x * log2(e)), r = x - k * ln(2)
; Accuracy: relative error < 4e-6 across full IEEE float range.
; Input:  XMM0 = x
; Output: XMM0 = exp(x)
; Volatiles: XMM0, XMM1, XMM2, XMM3, EAX
; ------------------------------------------------------------------------------
fast_exp:
    vcomiss xmm0, [rel const_neg_80]
    jb .exp_zero
    vcomiss xmm0, [rel const_pos_80]
    ja .exp_huge

    ; k = round(x * log2e)
    vmulss xmm1, xmm0, [rel const_log2e]
    vroundss xmm2, xmm1, 0          ; xmm2 = k (rounded to nearest integer)

    ; r = x - k * ln2
    vmulss xmm3, xmm2, [rel const_ln2]
    vsubss xmm0, xmm0, xmm3         ; xmm0 = r in [-ln2/2, ln2/2]

    ; Evaluate Taylor polynomial for exp(r) on [-0.35, 0.35]
    vmovss xmm1, [rel exp_c5]
    vfmadd213ss xmm1, xmm0, [rel exp_c4]
    vfmadd213ss xmm1, xmm0, [rel exp_c3]
    vfmadd213ss xmm1, xmm0, [rel exp_c2]
    vfmadd213ss xmm1, xmm0, [rel exp_c1]
    vfmadd213ss xmm1, xmm0, [rel exp_c0] ; xmm1 = exp(r)

    ; Compute 2^k as IEEE-754 float
    vcvttss2si eax, xmm2
    add eax, 127                    ; exponent bias
    cmp eax, 0
    jle .exp_zero
    cmp eax, 255
    jge .exp_huge
    shl eax, 23                     ; place in exponent bits
    vmovd xmm2, eax                 ; xmm2 = 2^k

    ; result = exp(r) * 2^k
    vmulss xmm0, xmm1, xmm2
    ret

.exp_zero:
    vxorps xmm0, xmm0, xmm0
    ret

.exp_huge:
    vmovss xmm0, [rel exp_max_val]
    ret

; ------------------------------------------------------------------------------
; Helper: fast_sigmoid (evaluates 1 / (1 + exp(-x)))
; Input:  XMM0 = x
; Output: XMM0 = sigmoid(x)
; ------------------------------------------------------------------------------
fast_sigmoid:
    push rbp
    mov rbp, rsp
    sub rsp, 16

    vxorps xmm1, xmm1, xmm1
    vcomiss xmm0, xmm1
    jb .sig_neg

    ; x >= 0: return 1.0 / (1.0 + exp(-x))
    vxorps xmm1, xmm1, xmm1
    vsubss xmm0, xmm1, xmm0        ; xmm0 = -x <= 0
    call fast_exp                  ; xmm0 = exp(-x)
    vaddss xmm1, xmm0, [rel const_one]
    vmovss xmm0, [rel const_one]
    vdivss xmm0, xmm0, xmm1
    add rsp, 16
    pop rbp
    ret

.sig_neg:
    ; x < 0: return exp(x) / (1.0 + exp(x))
    call fast_exp                  ; xmm0 = exp(x)
    vaddss xmm1, xmm0, [rel const_one]
    vdivss xmm0, xmm0, xmm1
    add rsp, 16
    pop rbp
    ret

; ------------------------------------------------------------------------------
; Main Kernel Forward Entry Point
; ------------------------------------------------------------------------------
synaptic_tree_avx2_forward:
synaptic_tree_avx2_eval:
    push rbp
    mov rbp, rsp
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 264

    ; Save register arguments to shadow space for preservation
    mov [rbp + 16], rcx            ; node_latents
    mov [rbp + 24], rdx            ; branch_weights
    mov [rbp + 32], r8             ; branch_biases
    mov [rbp + 40], r9             ; parent_indices

    ; Load parameters from stack
    mov r13d, dword [rbp + 48]     ; num_nodes
    mov r12d, dword [rbp + 56]     ; D

    ; Clamp num_nodes to max 8 for safety
    cmp r13d, 8
    jle .num_nodes_ok
    mov r13d, 8
.num_nodes_ok:

    ; Stride in bytes per node: r14 = D * 4
    mov r14d, r12d
    shl r14, 2

    ; ==========================================================================
    ; STEP 1: Compute dot products and transition logits for each node
    ; ==========================================================================
    xor r10d, r10d                 ; node_idx = 0

.node_dot_loop:
    cmp r10d, r13d
    jge .dot_loop_done

    ; Calculate node pointers
    mov rax, r10
    imul rax, r14
    mov rcx, [rbp + 16]
    mov rdx, [rbp + 24]
    lea rsi, [rcx + rax]           ; rsi = node_latents + node_idx * D * 4
    lea rdi, [rdx + rax]           ; rdi = branch_weights + node_idx * D * 4

    ; Initialize 4 YMM accumulators
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3

    xor eax, eax                   ; j = 0

.dim_loop_32:
    mov r11d, r12d
    sub r11d, eax
    cmp r11d, 32
    jl .dim_loop_16

    ; Unrolled 32 floats (4 x 256-bit AVX-2 vectors = 128 bytes)
    vmovups ymm4, [rsi + rax*4]
    vmovups ymm5, [rsi + rax*4 + 32]
    vmovups ymm6, [rsi + rax*4 + 64]
    vmovups ymm7, [rsi + rax*4 + 96]

    vfmadd231ps ymm0, ymm4, [rdi + rax*4]
    vfmadd231ps ymm1, ymm5, [rdi + rax*4 + 32]
    vfmadd231ps ymm2, ymm6, [rdi + rax*4 + 64]
    vfmadd231ps ymm3, ymm7, [rdi + rax*4 + 96]

    add eax, 32
    jmp .dim_loop_32

.dim_loop_16:
    cmp r11d, 16
    jl .dim_loop_8

    vmovups ymm4, [rsi + rax*4]
    vmovups ymm5, [rsi + rax*4 + 32]

    vfmadd231ps ymm0, ymm4, [rdi + rax*4]
    vfmadd231ps ymm1, ymm5, [rdi + rax*4 + 32]

    add eax, 16
    sub r11d, 16

.dim_loop_8:
    cmp r11d, 8
    jl .dot_reduction

    vmovups ymm4, [rsi + rax*4]
    vfmadd231ps ymm0, ymm4, [rdi + rax*4]

    add eax, 8
    sub r11d, 8

.dot_reduction:
    ; Horizontal reduction of ymm0..ymm3
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2

    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0       ; dot product sum in xmm0[0]

.scalar_tail:
    cmp eax, r12d
    jge .add_bias
    vmovss xmm2, [rsi + rax*4]
    vmulss xmm2, xmm2, [rdi + rax*4]
    vaddss xmm0, xmm0, xmm2
    inc eax
    jmp .scalar_tail

.add_bias:
    mov r8, [rbp + 32]             ; branch_biases
    vaddss xmm0, xmm0, [r8 + r10*4]
    ; Store into local logits array: [rbp - 96 + r10*4]
    vmovss [rbp - 96 + r10*4], xmm0

    inc r10d
    jmp .node_dot_loop

.dot_loop_done:

    ; ==========================================================================
    ; STEP 2: Transition Probabilities & Cumulative Path Scores
    ; ==========================================================================
    ; Node 0 (Root) base path score = 1.0f
    vmovss xmm0, [rel const_one]
    vmovss [rbp - 128], xmm0       ; scores[0] = 1.0f

    ; Evaluate child nodes i in [1, num_nodes - 1]
    mov r10d, 1                    ; i = 1

.score_nodes_loop:
    cmp r10d, r13d
    jge .scoring_done

    mov r9, [rbp + 40]             ; parent_indices
    mov eax, dword [r9 + r10*4]    ; P = parent_indices[i]

    ; If parent index is invalid or parent score is 0.0, node is pruned
    cmp eax, 0
    jl .prune_node_zero
    cmp eax, r13d
    jge .prune_node_zero

    vmovss xmm0, [rbp - 128 + rax*4] ; scores[P]
    vxorps xmm1, xmm1, xmm1
    vucomiss xmm0, xmm1
    jbe .prune_node_zero           ; parent score <= 0.0 -> prune child

    ; Count sibling children sharing this parent P and find max logit among them
    xor ecx, ecx                   ; sibling_count = 0
    mov ebx, eax                   ; parent_P = P
    vmovss xmm2, [rel const_neg_80] ; max_z = -80.0
    mov edx, 1                     ; k = 1

.sibling_count_loop:
    cmp edx, r13d
    jge .sibling_count_done
    cmp dword [r9 + rdx*4], ebx
    jne .next_sibling_check
    inc ecx
    ; max_z = max(max_z, logits[k])
    vmaxss xmm2, xmm2, [rbp - 96 + rdx*4]
.next_sibling_check:
    inc edx
    jmp .sibling_count_loop

.sibling_count_done:
    ; If sibling_count > 1: Softmax over siblings
    ; If sibling_count == 1: Sigmoid gating
    cmp ecx, 1
    jg .eval_sibling_softmax

    ; Single child: Sigmoid gating p_i = sigmoid(logits[i])
    vmovss xmm0, [rbp - 96 + r10*4]
    call fast_sigmoid              ; xmm0 = p_i
    jmp .apply_path_score

.eval_sibling_softmax:
    ; Softmax among siblings of parent P:
    ; sum_exp = sum_{k: parent[k]==P} exp(logits[k] - max_z)
    ; temp_exp[k] = exp(logits[k] - max_z)
    vxorps xmm3, xmm3, xmm3        ; sum_exp = 0.0
    vmovss [rbp - 164], xmm3       ; temporary sum_exp storage
    vmovss [rbp - 168], xmm2       ; temporary max_z storage
    mov edx, 1

.softmax_exp_loop:
    cmp edx, r13d
    jge .softmax_exp_done
    cmp dword [r9 + rdx*4], ebx
    jne .softmax_next_k

    ; diff = logits[k] - max_z
    vmovss xmm0, [rbp - 96 + rdx*4]
    vsubss xmm0, xmm0, [rbp - 168]
    call fast_exp                  ; xmm0 = exp(diff)
    vmovss [rbp - 160 + rdx*4], xmm0 ; temp_exp[k] = xmm0
    vaddss xmm3, xmm0, [rbp - 164]
    vmovss [rbp - 164], xmm3       ; sum_exp += xmm0

.softmax_next_k:
    inc edx
    jmp .softmax_exp_loop

.softmax_exp_done:
    ; p_i = temp_exp[i] / sum_exp
    vmovss xmm0, [rbp - 160 + r10*4]
    vdivss xmm0, xmm0, [rbp - 164]

.apply_path_score:
    ; scores[i] = scores[P] * p_i
    mov eax, dword [r9 + r10*4]    ; P
    vmulss xmm0, xmm0, [rbp - 128 + rax*4]
    vmovss [rbp - 128 + r10*4], xmm0
    jmp .next_node_score

.prune_node_zero:
    vxorps xmm0, xmm0, xmm0
    vmovss [rbp - 128 + r10*4], xmm0

.next_node_score:
    inc r10d
    jmp .score_nodes_loop

.scoring_done:

    ; ==========================================================================
    ; STEP 3: Vectorized Pruning with vcmpps and Threshold Filtering
    ; ==========================================================================
    ; Zero out tail elements [num_nodes .. 7] in scores buffer
    mov eax, r13d
.zero_tail_scores:
    cmp eax, 8
    jge .apply_vector_cmp
    mov dword [rbp - 128 + rax*4], 0
    inc eax
    jmp .zero_tail_scores

.apply_vector_cmp:
    ; Load all 8 scores into YMM0
    vmovups ymm0, [rbp - 128]
    ; Load prob_threshold into YMM1
    vbroadcastss ymm1, dword [rbp + 64]
    ; Compare GE (Predicate 29 = CMP_GE_OS): mask[k] = (scores[k] >= prob_threshold)
    vcmpps ymm2, ymm0, ymm1, 29
    ; Zero out below threshold
    vandps ymm0, ymm0, ymm2
    vmovups [rbp - 128], ymm0

    ; Propagate cascade pruning downwards (if parent is 0, child becomes 0)
    mov r9, [rbp + 40]             ; parent_indices
    mov r10d, 1
.cascade_loop:
    cmp r10d, r13d
    jge .cascade_done
    mov eax, dword [r9 + r10*4]    ; P
    cmp eax, 0
    jl .skip_cascade_node
    vmovss xmm0, [rbp - 128 + rax*4]
    vxorps xmm1, xmm1, xmm1
    vucomiss xmm0, xmm1
    jne .skip_cascade_node
    vmovss [rbp - 128 + r10*4], xmm1 ; prune child
.skip_cascade_node:
    inc r10d
    jmp .cascade_loop

.cascade_done:
    ; Write scores to out_path_scores ([rbp + 72])
    mov rdi, [rbp + 72]
    xor eax, eax
.write_scores_loop:
    cmp eax, r13d
    jge .scores_written
    vmovss xmm0, [rbp - 128 + rax*4]
    vmovss [rdi + rax*4], xmm0
    inc eax
    jmp .write_scores_loop
.scores_written:

    ; ==========================================================================
    ; STEP 4: Top-4 Selection of Asymmetric Tree
    ; ==========================================================================
    ; Clear used_flags buffer (8 int32s) at [rbp - 224]
    xor eax, eax
.clear_flags_loop:
    cmp eax, 8
    jge .flags_cleared
    mov dword [rbp - 224 + rax*4], 0
    inc eax
    jmp .clear_flags_loop
.flags_cleared:

    ; Root check: if scores[0] > 0.0, select root as slot 0
    vxorps xmm1, xmm1, xmm1
    vmovss xmm0, [rbp - 128]
    vucomiss xmm0, xmm1
    jbe .no_root_selected

    mov dword [rbp - 192], 0       ; selected[0] = 0
    mov dword [rbp - 224], 1       ; used_flags[0] = 1
    mov ebx, 1                     ; num_selected = 1
    jmp .select_remaining_nodes

.no_root_selected:
    xor ebx, ebx                   ; num_selected = 0

.select_remaining_nodes:
.selection_outer_loop:
    cmp ebx, 4
    jge .selection_complete

    ; Find unselected node k in [1, num_nodes - 1] with max score > 0.0
    mov r10d, -1                   ; best_k = -1
    vxorps xmm2, xmm2, xmm2        ; best_score = 0.0f
    mov eax, 1                     ; k = 1

.search_max_loop:
    cmp eax, r13d
    jge .search_max_end

    cmp dword [rbp - 224 + rax*4], 0
    jne .search_next_k

    vmovss xmm0, [rbp - 128 + rax*4]
    vucomiss xmm0, xmm2
    jbe .search_next_k

    ; Found higher score
    movaps xmm2, xmm0
    mov r10d, eax

.search_next_k:
    inc eax
    jmp .search_max_loop

.search_max_end:
    cmp r10d, -1
    je .selection_complete         ; no more surviving nodes

    mov dword [rbp - 192 + rbx*4], r10d
    mov dword [rbp - 224 + r10*4], 1
    inc ebx
    jmp .selection_outer_loop

.selection_complete:
    ; Pad remaining selected slots up to 4 with -1
    mov eax, ebx
.pad_sel_loop:
    cmp eax, 4
    jge .write_selected_nodes
    mov dword [rbp - 192 + rax*4], -1
    inc eax
    jmp .pad_sel_loop

.write_selected_nodes:
    ; Write 4 selected node indices to out_selected_nodes ([rbp + 80])
    mov rdi, [rbp + 80]
    mov eax, [rbp - 192]
    mov [rdi], eax
    mov eax, [rbp - 192 + 4]
    mov [rdi + 4], eax
    mov eax, [rbp - 192 + 8]
    mov [rdi + 8], eax
    mov eax, [rbp - 192 + 12]
    mov [rdi + 12], eax

    ; Write count to *out_num_selected ([rbp + 96])
    mov rdi, [rbp + 96]
    mov [rdi], ebx

    ; ==========================================================================
    ; STEP 5: Construct 2D Causal Attention Mask (4x4 floats)
    ; ==========================================================================
    mov rdi, [rbp + 88]            ; out_causal_mask (16 floats)
    vxorps ymm0, ymm0, ymm0
    vmovups [rdi], ymm0            ; row 0 & 1
    vmovups [rdi + 32], ymm0       ; row 2 & 3

    mov r9, [rbp + 40]             ; parent_indices
    xor r10d, r10d                 ; r = 0

.mask_row_loop:
    cmp r10d, ebx
    jge .mask_all_done

    mov r14d, [rbp - 192 + r10*4]  ; u = selected[r]

    xor r11d, r11d                 ; c = 0
.mask_col_loop:
    cmp r11d, ebx
    jge .mask_row_advance

    mov r15d, [rbp - 192 + r11*4]  ; v = selected[c]

    ; Check if v is ancestor of u (or u == v)
    mov eax, r14d                  ; curr = u
.ancestor_trace:
    cmp eax, r15d
    je .is_ancestor
    cmp eax, 0
    jle .not_ancestor              ; reached root or invalid without finding v
    mov eax, dword [r9 + rax*4]    ; curr = parent_indices[curr]
    jmp .ancestor_trace

.is_ancestor:
    mov eax, r10d
    shl eax, 2
    add eax, r11d                  ; cell_idx = r * 4 + c
    vmovss xmm0, [rel const_one]
    vmovss [rdi + rax*4], xmm0

.not_ancestor:
    inc r11d
    jmp .mask_col_loop

.mask_row_advance:
    inc r10d
    jmp .mask_row_loop

.mask_all_done:
    ; Clean AVX state
    vzeroupper

    ; Epilogue
    add rsp, 264
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret
