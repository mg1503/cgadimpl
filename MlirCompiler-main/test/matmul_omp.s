	.text
	.file	"matmul_omp.cpp"
	.globl	_Z15matmul_parallelPKfS0_Pf     # -- Begin function _Z15matmul_parallelPKfS0_Pf
	.p2align	4, 0x90
	.type	_Z15matmul_parallelPKfS0_Pf,@function
_Z15matmul_parallelPKfS0_Pf:            # @_Z15matmul_parallelPKfS0_Pf
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	24(%rsp), %r14
	leaq	20(%rsp), %rbx
	leaq	.L__unnamed_1(%rip), %r15
	movq	%rdi, 40(%rsp)
	movq	%rsi, 32(%rsp)
	movq	%rdx, 24(%rsp)
	leaq	.omp_outlined.(%rip), %rdx
	movl	$2, %esi
	xorl	%eax, %eax
	movl	$512, 20(%rsp)                  # imm = 0x200
	movq	%r15, %rdi
	movq	%r14, %rcx
	movq	%rbx, %r8
	callq	__kmpc_fork_call@PLT
	leaq	32(%rsp), %rax
	leaq	.omp_outlined..1(%rip), %rdx
	leaq	40(%rsp), %r8
	movq	%r15, %rdi
	movl	$4, %esi
	movq	%rbx, %rcx
	movq	%r14, %r9
	movq	%rax, (%rsp)
	xorl	%eax, %eax
	callq	__kmpc_fork_call@PLT
	addq	$48, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	_Z15matmul_parallelPKfS0_Pf, .Lfunc_end0-_Z15matmul_parallelPKfS0_Pf
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function .omp_outlined.
	.type	.omp_outlined.,@function
.omp_outlined.:                         # @.omp_outlined.
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movl	(%rdi), %ebx
	movq	%rdx, %r14
	movl	$0, 12(%rsp)
	movl	$262143, 8(%rsp)                # imm = 0x3FFFF
	movl	$1, 20(%rsp)
	movl	$0, 16(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	28(%rsp), %rax
	leaq	24(%rsp), %rcx
	leaq	.L__unnamed_2(%rip), %rdi
	leaq	20(%rsp), %r8
	leaq	16(%rsp), %r9
	movl	%ebx, %esi
	movl	$34, %edx
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	__kmpc_for_static_init_4@PLT
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	movl	8(%rsp), %ecx
	movl	$262143, %eax                   # imm = 0x3FFFF
	cmpl	$262143, %ecx                   # imm = 0x3FFFF
	cmovll	%ecx, %eax
	movslq	12(%rsp), %rcx
	movl	%eax, 8(%rsp)
	cmpl	%ecx, %eax
	jl	.LBB1_2
# %bb.1:
	movq	%rcx, %rdi
	subl	%ecx, %eax
	xorl	%esi, %esi
	shlq	$2, %rdi
	addq	(%r14), %rdi
	leaq	4(,%rax,4), %rdx
	callq	memset@PLT
.LBB1_2:
	leaq	.L__unnamed_2(%rip), %rdi
	movl	%ebx, %esi
	callq	__kmpc_for_static_fini@PLT
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	.omp_outlined., .Lfunc_end1-.omp_outlined.
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function .omp_outlined..1
	.type	.omp_outlined..1,@function
.omp_outlined..1:                       # @.omp_outlined..1
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$184, %rsp
	.cfi_def_cfa_offset 240
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	(%rdi), %ebx
	leaq	.L__unnamed_1(%rip), %r12
	movq	%r9, 152(%rsp)                  # 8-byte Spill
	movq	%r8, 144(%rsp)                  # 8-byte Spill
	movq	%rcx, 120(%rsp)                 # 8-byte Spill
	movl	$1073741859, %edx               # imm = 0x40000023
	xorl	%ecx, %ecx
	movl	$63, %r8d
	movl	$1, %r9d
	movl	$0, 20(%rsp)
	movl	$63, 16(%rsp)
	movl	$1, 36(%rsp)
	movl	$0, 32(%rsp)
	movl	$1, (%rsp)
	movq	%r12, %rdi
	movl	%ebx, %esi
	callq	__kmpc_dispatch_init_4@PLT
	leaq	32(%rsp), %rdx
	leaq	20(%rsp), %rcx
	leaq	16(%rsp), %r8
	leaq	36(%rsp), %r9
	movq	%r12, %rdi
	movl	%ebx, %esi
	movl	%ebx, 56(%rsp)                  # 4-byte Spill
	jmp	.LBB2_1
	.p2align	4, 0x90
.LBB2_19:                               #   in Loop: Header=BB2_1 Depth=1
	movl	56(%rsp), %esi                  # 4-byte Reload
	leaq	.L__unnamed_1(%rip), %rdi
	leaq	32(%rsp), %rdx
	leaq	20(%rsp), %rcx
	leaq	16(%rsp), %r8
	leaq	36(%rsp), %r9
.LBB2_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_4 Depth 2
                                        #       Child Loop BB2_8 Depth 3
                                        #         Child Loop BB2_9 Depth 4
                                        #           Child Loop BB2_10 Depth 5
                                        #             Child Loop BB2_14 Depth 6
                                        #             Child Loop BB2_12 Depth 6
	vzeroupper
	callq	__kmpc_dispatch_next_4@PLT
	testl	%eax, %eax
	je	.LBB2_20
# %bb.2:                                #   in Loop: Header=BB2_1 Depth=1
	movl	20(%rsp), %eax
	movl	16(%rsp), %ecx
	cmpl	%ecx, %eax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movl	%ecx, 64(%rsp)                  # 4-byte Spill
	jg	.LBB2_19
# %bb.3:                                #   in Loop: Header=BB2_1 Depth=1
	movq	88(%rsp), %rax                  # 8-byte Reload
	movl	$64, %esi
	movl	$63, %edx
	movl	%eax, %ecx
	shll	$6, %ecx
	subl	%ecx, %esi
	subl	%ecx, %edx
	movl	%ecx, 28(%rsp)                  # 4-byte Spill
	xorl	%ecx, %ecx
	movl	%esi, 24(%rsp)                  # 4-byte Spill
	movl	%eax, %esi
	movl	%edx, 60(%rsp)                  # 4-byte Spill
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	jmp	.LBB2_4
	.p2align	4, 0x90
.LBB2_18:                               #   in Loop: Header=BB2_4 Depth=2
	movq	48(%rsp), %rax                  # 8-byte Reload
	addl	$64, 28(%rsp)                   # 4-byte Folded Spill
	addl	$-64, 24(%rsp)                  # 4-byte Folded Spill
	addl	$1, %eax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	112(%rsp), %rax                 # 8-byte Reload
	cmpl	64(%rsp), %eax                  # 4-byte Folded Reload
	leal	1(%rax), %eax
	movl	%eax, %esi
	je	.LBB2_19
.LBB2_4:                                #   Parent Loop BB2_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_8 Depth 3
                                        #         Child Loop BB2_9 Depth 4
                                        #           Child Loop BB2_10 Depth 5
                                        #             Child Loop BB2_14 Depth 6
                                        #             Child Loop BB2_12 Depth 6
	leal	7(%rsi), %edx
	testl	%esi, %esi
	movq	%rsi, 112(%rsp)                 # 8-byte Spill
	movl	$448, %ebx                      # imm = 0x1C0
	cmovnsl	%esi, %edx
                                        # kill: def $esi killed $esi killed $rsi def $rsi
	movl	%edx, %eax
	andl	$-8, %edx
	sarl	$3, %eax
	subl	%edx, %esi
	movl	$448, %edx                      # imm = 0x1C0
	movl	%eax, %ecx
	shll	$6, %esi
	shll	$6, %ecx
	movq	%rsi, 40(%rsp)                  # 8-byte Spill
	cmpl	$448, %ecx                      # imm = 0x1C0
	cmovll	%ecx, %edx
	addl	$64, %edx
	cmpl	$448, %esi                      # imm = 0x1C0
	cmovll	%esi, %ebx
	cmpl	%edx, %ecx
	jge	.LBB2_18
# %bb.5:                                #   in Loop: Header=BB2_4 Depth=2
	leal	64(%rbx), %edi
	cmpl	40(%rsp), %edi                  # 4-byte Folded Reload
	jle	.LBB2_18
# %bb.6:                                #   in Loop: Header=BB2_4 Depth=2
	subl	40(%rsp), %edi                  # 4-byte Folded Reload
	je	.LBB2_18
# %bb.7:                                #   in Loop: Header=BB2_4 Depth=2
	movq	88(%rsp), %rsi                  # 8-byte Reload
	movq	48(%rsp), %r8                   # 8-byte Reload
	movl	60(%rsp), %r9d                  # 4-byte Reload
	movslq	%ecx, %rcx
	movq	%rcx, 128(%rsp)                 # 8-byte Spill
	movslq	%edx, %rcx
	imull	$32256, %eax, %edx              # imm = 0x7E00
	shll	$9, %eax
	movq	%rcx, 160(%rsp)                 # 8-byte Spill
	movq	40(%rsp), %rcx                  # 8-byte Reload
	leal	(%rsi,%r8), %ebp
	movl	%r8d, %esi
	shll	$6, %esi
	shll	$6, %ebp
	subl	%esi, %r9d
	addl	%edx, %ebp
	addl	%r9d, %ebx
	movl	%ebp, 76(%rsp)                  # 4-byte Spill
	addl	%eax, %ebx
	addl	%ebx, %ecx
	movq	%rbx, 168(%rsp)                 # 8-byte Spill
	movl	%ecx, 72(%rsp)                  # 4-byte Spill
	movl	28(%rsp), %ecx                  # 4-byte Reload
	movl	%ecx, %r9d
	addl	%ecx, %edx
	movl	$448, %ecx                      # imm = 0x1C0
	subl	%eax, %r9d
	movl	%edx, 68(%rsp)                  # 4-byte Spill
	cmpl	$448, %r9d                      # imm = 0x1C0
	cmovll	%r9d, %ecx
	addl	24(%rsp), %ecx                  # 4-byte Folded Reload
	addl	%eax, %ecx
	movl	$64, %eax
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%ecx, 84(%rsp)                  # 4-byte Spill
	jmp	.LBB2_8
	.p2align	4, 0x90
.LBB2_17:                               #   in Loop: Header=BB2_8 Depth=3
	addq	$64, 104(%rsp)                  # 8-byte Folded Spill
	cmpl	$448, 96(%rsp)                  # 4-byte Folded Reload
                                        # imm = 0x1C0
	movq	136(%rsp), %rax                 # 8-byte Reload
	jae	.LBB2_18
.LBB2_8:                                #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_4 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB2_9 Depth 4
                                        #           Child Loop BB2_10 Depth 5
                                        #             Child Loop BB2_14 Depth 6
                                        #             Child Loop BB2_12 Depth 6
	movq	120(%rsp), %rsi                 # 8-byte Reload
	movl	68(%rsp), %edx                  # 4-byte Reload
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	addq	$64, %rax
	xorl	%ecx, %ecx
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	movl	%edx, %r14d
	movq	(%rsi), %r13
	movq	128(%rsp), %rdx                 # 8-byte Reload
	jmp	.LBB2_9
	.p2align	4, 0x90
.LBB2_16:                               #   in Loop: Header=BB2_9 Depth=4
	movq	176(%rsp), %rdx                 # 8-byte Reload
	movl	80(%rsp), %ecx                  # 4-byte Reload
	addl	$512, %r14d                     # imm = 0x200
	addq	$1, %rdx
	addl	$1, %ecx
	cmpq	160(%rsp), %rdx                 # 8-byte Folded Reload
	je	.LBB2_17
.LBB2_9:                                #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_4 Depth=2
                                        #       Parent Loop BB2_8 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB2_10 Depth 5
                                        #             Child Loop BB2_14 Depth 6
                                        #             Child Loop BB2_12 Depth 6
	movl	72(%rsp), %eax                  # 4-byte Reload
	cmpl	40(%rsp), %eax                  # 4-byte Folded Reload
	movl	%ecx, %esi
	movl	%ecx, 80(%rsp)                  # 4-byte Spill
	movq	152(%rsp), %rax                 # 8-byte Reload
	movq	%rdx, 176(%rsp)                 # 8-byte Spill
	movl	%edx, %ecx
	movq	168(%rsp), %rdx                 # 8-byte Reload
	movq	(%rax), %rbx
	movq	144(%rsp), %rax                 # 8-byte Reload
	setl	%r8b
	shll	$9, %esi
	addl	76(%rsp), %esi                  # 4-byte Folded Reload
	shll	$9, %ecx
	movslq	%ecx, %rcx
	movq	(%rax), %rax
	addl	%esi, %edx
	cmpl	%esi, %edx
	setl	%r12b
	orb	%r8b, %r12b
	movq	96(%rsp), %r8                   # 8-byte Reload
	jmp	.LBB2_10
	.p2align	4, 0x90
.LBB2_15:                               #   in Loop: Header=BB2_10 Depth=5
	addq	$1, %r8
	cmpq	104(%rsp), %r8                  # 8-byte Folded Reload
	je	.LBB2_16
.LBB2_10:                               #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_4 Depth=2
                                        #       Parent Loop BB2_8 Depth=3
                                        #         Parent Loop BB2_9 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB2_14 Depth 6
                                        #             Child Loop BB2_12 Depth 6
	testl	%edi, %edi
	leaq	(%r8,%rcx), %rsi
	sete	%dl
	vmovss	(%r13,%rsi,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%r8, %rsi
	orb	%r12b, %dl
	shlq	$9, %rsi
	cmpb	$1, %dl
	jne	.LBB2_13
# %bb.11:                               #   in Loop: Header=BB2_10 Depth=5
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB2_12:                               #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_4 Depth=2
                                        #       Parent Loop BB2_8 Depth=3
                                        #         Parent Loop BB2_9 Depth=4
                                        #           Parent Loop BB2_10 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	leal	(%r9,%rdx), %ebp
	movslq	%ebp, %rbp
	addq	%rsi, %rbp
	vmovss	(%rbx,%rbp,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	leal	(%r14,%rdx), %ebp
	addl	$1, %edx
	movslq	%ebp, %rbp
	cmpl	%edi, %edx
	vfmadd213ss	(%rax,%rbp,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%rax,%rbp,4)
	jb	.LBB2_12
	jmp	.LBB2_15
	.p2align	4, 0x90
.LBB2_13:                               #   in Loop: Header=BB2_10 Depth=5
	movl	84(%rsp), %edx                  # 4-byte Reload
	vbroadcastss	%xmm0, %ymm0
	movl	%r9d, %r11d
	movl	%r14d, %r15d
	.p2align	4, 0x90
.LBB2_14:                               #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_4 Depth=2
                                        #       Parent Loop BB2_8 Depth=3
                                        #         Parent Loop BB2_9 Depth=4
                                        #           Parent Loop BB2_10 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movslq	%r11d, %r11
	movslq	%r15d, %r15
	leaq	(%rsi,%r11), %r10
	addl	$32, %r11d
	vmovups	(%rbx,%r10,4), %ymm1
	vmovups	32(%rbx,%r10,4), %ymm2
	vmovups	64(%rbx,%r10,4), %ymm3
	vmovups	96(%rbx,%r10,4), %ymm4
	vfmadd213ps	(%rax,%r15,4), %ymm0, %ymm1 # ymm1 = (ymm0 * ymm1) + mem
	vfmadd213ps	32(%rax,%r15,4), %ymm0, %ymm2 # ymm2 = (ymm0 * ymm2) + mem
	vfmadd213ps	64(%rax,%r15,4), %ymm0, %ymm3 # ymm3 = (ymm0 * ymm3) + mem
	vfmadd213ps	96(%rax,%r15,4), %ymm0, %ymm4 # ymm4 = (ymm0 * ymm4) + mem
	vmovups	%ymm1, (%rax,%r15,4)
	vmovups	%ymm2, 32(%rax,%r15,4)
	vmovups	%ymm3, 64(%rax,%r15,4)
	vmovups	%ymm4, 96(%rax,%r15,4)
	addl	$32, %r15d
	addl	$-32, %edx
	jne	.LBB2_14
	jmp	.LBB2_15
.LBB2_20:
	addq	$184, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	.omp_outlined..1, .Lfunc_end2-.omp_outlined..1
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function main
.LCPI3_0:
	.long	0x3f800000                      # float 1
.LCPI3_1:
	.long	0x40000000                      # float 2
.LCPI3_2:
	.long	0x44800000                      # float 1024
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3
.LCPI3_3:
	.quad	0x4090000000000000              # double 1024
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$64, %edi
	movl	$1048576, %esi                  # imm = 0x100000
	callq	aligned_alloc@PLT
	movl	$64, %edi
	movl	$1048576, %esi                  # imm = 0x100000
	movq	%rax, %r14
	callq	aligned_alloc@PLT
	movl	$64, %edi
	movl	$1048576, %esi                  # imm = 0x100000
	movq	%rax, %rbx
	callq	aligned_alloc@PLT
	testq	%r14, %r14
	je	.LBB3_9
# %bb.1:
	testq	%rbx, %rbx
	je	.LBB3_9
# %bb.2:
	movq	%rax, %r15
	testq	%rax, %rax
	je	.LBB3_9
# %bb.3:
	vbroadcastss	.LCPI3_0(%rip), %ymm0   # ymm0 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vbroadcastss	.LCPI3_1(%rip), %ymm1   # ymm1 = [2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0]
	movq	$-1048576, %rax                 # imm = 0xFFF00000
	.p2align	4, 0x90
.LBB3_4:                                # =>This Inner Loop Header: Depth=1
	vmovaps	%ymm0, 1048576(%r14,%rax)
	vmovaps	%ymm0, 1048608(%r14,%rax)
	vmovaps	%ymm0, 1048640(%r14,%rax)
	vmovaps	%ymm1, 1048576(%rbx,%rax)
	vmovaps	%ymm1, 1048608(%rbx,%rax)
	vmovaps	%ymm1, 1048640(%rbx,%rax)
	vmovaps	%ymm0, 1048672(%r14,%rax)
	vmovaps	%ymm1, 1048672(%rbx,%rax)
	subq	$-128, %rax
	jne	.LBB3_4
# %bb.5:
	leaq	.L__unnamed_1(%rip), %rbp
	leaq	16(%rsp), %r13
	leaq	12(%rsp), %r12
	leaq	.omp_outlined.(%rip), %rdx
	movl	$2, %esi
	xorl	%eax, %eax
	movq	%r14, 32(%rsp)
	movq	%rbx, 24(%rsp)
	movq	%r15, 16(%rsp)
	movl	$512, 12(%rsp)                  # imm = 0x200
	movq	%rbp, %rdi
	movq	%r13, %rcx
	movq	%r12, %r8
	vzeroupper
	callq	__kmpc_fork_call@PLT
	leaq	24(%rsp), %rax
	movq	%rbp, %rdi
	leaq	.omp_outlined..1(%rip), %rdx
	leaq	32(%rsp), %r8
	movl	$4, %esi
	movq	%r12, %rcx
	movq	%r13, %r9
	xorl	%ebp, %ebp
	movq	%rax, (%rsp)
	xorl	%eax, %eax
	callq	__kmpc_fork_call@PLT
	vmovss	(%r15), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	vmovss	.LCPI3_2(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vucomiss	%xmm0, %xmm1
	jne	.LBB3_6
.LBB3_7:
	movq	%r14, %rdi
	callq	free@PLT
	movq	%rbx, %rdi
	callq	free@PLT
	movq	%r15, %rdi
	callq	free@PLT
.LBB3_8:
	movl	%ebp, %eax
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB3_9:
	.cfi_def_cfa_offset 96
	movq	stderr@GOTPCREL(%rip), %rax
	leaq	.L.str(%rip), %rdi
	movl	$25, %esi
	movl	$1, %edx
	movq	(%rax), %rcx
	callq	fwrite@PLT
	movl	$1, %ebp
	jmp	.LBB3_8
.LBB3_6:
	movq	stderr@GOTPCREL(%rip), %rax
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vmovsd	.LCPI3_3(%rip), %xmm1           # xmm1 = mem[0],zero
	leaq	.L.str.2(%rip), %rsi
	movq	(%rax), %rdi
	movb	$2, %al
	callq	fprintf@PLT
	movl	$1, %ebp
	jmp	.LBB3_7
.Lfunc_end3:
	.size	main, .Lfunc_end3-main
	.cfi_endproc
                                        # -- End function
	.type	.L__unnamed_3,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_3:
	.asciz	";unknown;unknown;0;0;;"
	.size	.L__unnamed_3, 23

	.type	.L__unnamed_2,@object           # @1
	.section	.data.rel.ro,"aw",@progbits
	.p2align	3
.L__unnamed_2:
	.long	0                               # 0x0
	.long	514                             # 0x202
	.long	0                               # 0x0
	.long	22                              # 0x16
	.quad	.L__unnamed_3
	.size	.L__unnamed_2, 24

	.type	.L__unnamed_1,@object           # @2
	.p2align	3
.L__unnamed_1:
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	22                              # 0x16
	.quad	.L__unnamed_3
	.size	.L__unnamed_1, 24

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Memory allocation failed\n"
	.size	.L.str, 26

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"Result incorrect: got %f, expected %f\n"
	.size	.L.str.2, 39

	.ident	"Ubuntu clang version 14.0.0-1ubuntu1.1"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym .omp_outlined.
	.addrsig_sym .omp_outlined..1
