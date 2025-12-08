module {
  memref.global "private" constant @__constant_4096x4096xf32_0 : memref<4096x4096xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_4096x4096xf32 : memref<4096x4096xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func private @get_time() -> i64
  func.func private @print_gflops(i64, i64, i64, i64)
  func.func @core_op(%arg0: memref<4096x4096xf32, strided<[?, ?], offset: ?>>, %arg1: memref<4096x4096xf32, strided<[?, ?], offset: ?>>) -> memref<4096x4096xf32> {
    %0 = ub.poison : vector<16x8xf32>
    %1 = ub.poison : vector<8xf32>
    %2 = ub.poison : vector<1x8xf32>
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %3 = ub.poison : vector<1xf32>
    %4 = ub.poison : vector<16x1xf32>
    %c2621440 = arith.constant 2621440 : index
    %c1200 = arith.constant 1200 : index
    %c-1200 = arith.constant -1200 : index
    %c640 = arith.constant 640 : index
    %c-640 = arith.constant -640 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %5 = ub.poison : f32
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c500 = arith.constant 500 : index
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c4096_i64 = arith.constant 4096 : i64
    %6 = call @get_time() : () -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096x4096xf32>
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg2, %arg3) : index = (%c0, %c0) to (%c7, %c4) step (%c1, %c1) {
          %9 = arith.muli %arg2, %c-640 overflow<nsw> : index
          %10 = arith.addi %9, %c4096 : index
          %11 = arith.minsi %10, %c640 : index
          %12 = arith.muli %arg3, %c-1200 overflow<nsw> : index
          %13 = arith.addi %12, %c4096 : index
          %14 = arith.minsi %13, %c1200 : index
          cf.br ^bb1(%c0 : index)
        ^bb1(%15: index):  // 2 preds: ^bb0, ^bb127
          %16 = arith.cmpi slt, %15, %c4096 : index
          cf.cond_br %16, ^bb2, ^bb128
        ^bb2:  // pred: ^bb1
          %17 = arith.subi %c4096, %15 : index
          %18 = arith.minsi %17, %c500 : index
          cf.br ^bb3(%c0 : index)
        ^bb3(%19: index):  // 2 preds: ^bb2, ^bb126
          %20 = arith.cmpi slt, %19, %11 : index
          cf.cond_br %20, ^bb4, ^bb127
        ^bb4:  // pred: ^bb3
          cf.br ^bb5(%c0 : index)
        ^bb5(%21: index):  // 2 preds: ^bb4, ^bb125
          %22 = arith.cmpi slt, %21, %14 : index
          cf.cond_br %22, ^bb6, ^bb126
        ^bb6:  // pred: ^bb5
          %23 = arith.subi %11, %19 : index
          %24 = arith.minsi %23, %c16 : index
          %25 = arith.subi %14, %21 : index
          %26 = arith.minsi %25, %c8 : index
          %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<4096x4096xf32, strided<[?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index
          %27 = arith.muli %arg2, %strides#0 overflow<nsw> : index
          %28 = arith.muli %27, %c640 overflow<nsw> : index
          %29 = arith.addi %28, %offset : index
          %30 = arith.muli %15, %strides#1 overflow<nsw> : index
          %31 = arith.addi %29, %30 : index
          %32 = arith.muli %19, %strides#0 overflow<nsw> : index
          %33 = arith.addi %31, %32 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%33], sizes: [%24, %18], strides: [%strides#0, %strides#1] : memref<4096x4096xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<4096x4096xf32, strided<[?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index
          %34 = arith.muli %arg3, %strides_3#1 overflow<nsw> : index
          %35 = arith.muli %34, %c1200 overflow<nsw> : index
          %36 = arith.addi %35, %offset_1 : index
          %37 = arith.muli %15, %strides_3#0 overflow<nsw> : index
          %38 = arith.addi %36, %37 : index
          %39 = arith.muli %21, %strides_3#1 overflow<nsw> : index
          %40 = arith.addi %38, %39 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%40], sizes: [%18, %26], strides: [%strides_3#0, %strides_3#1] : memref<4096x4096xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %41 = arith.muli %arg2, %c2621440 overflow<nsw> : index
          %42 = arith.muli %arg3, %c1200 overflow<nsw> : index
          %43 = arith.addi %41, %42 : index
          %44 = arith.muli %19, %c4096 overflow<nsw> : index
          %45 = arith.addi %43, %44 : index
          %46 = arith.addi %45, %21 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %alloc to offset: [%46], sizes: [%24, %26], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %47 = vector.create_mask %24, %18 : vector<16x1xi1>
          %48 = vector.extract %47[0] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb7(%c0, %3 : index, vector<1xf32>)
        ^bb7(%49: index, %50: vector<1xf32>):  // 2 preds: ^bb6, ^bb12
          %51 = arith.cmpi slt, %49, %c1 : index
          cf.cond_br %51, ^bb8, ^bb13
        ^bb8:  // pred: ^bb7
          %52 = vector.extract %48[%49] : i1 from vector<1xi1>
          cf.cond_br %52, ^bb9, ^bb10
        ^bb9:  // pred: ^bb8
          %53 = memref.load %reinterpret_cast[%c0, %49] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %54 = vector.insert %53, %50 [%49] : f32 into vector<1xf32>
          cf.br ^bb11(%54 : vector<1xf32>)
        ^bb10:  // pred: ^bb8
          cf.br ^bb11(%50 : vector<1xf32>)
        ^bb11(%55: vector<1xf32>):  // 2 preds: ^bb9, ^bb10
          cf.br ^bb12
        ^bb12:  // pred: ^bb11
          %56 = arith.addi %49, %c1 : index
          cf.br ^bb7(%56, %55 : index, vector<1xf32>)
        ^bb13:  // pred: ^bb7
          %57 = vector.insert %50, %4 [0] : vector<1xf32> into vector<16x1xf32>
          %58 = vector.extract %47[1] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb14(%c0, %3 : index, vector<1xf32>)
        ^bb14(%59: index, %60: vector<1xf32>):  // 2 preds: ^bb13, ^bb19
          %61 = arith.cmpi slt, %59, %c1 : index
          cf.cond_br %61, ^bb15, ^bb20
        ^bb15:  // pred: ^bb14
          %62 = vector.extract %58[%59] : i1 from vector<1xi1>
          cf.cond_br %62, ^bb16, ^bb17
        ^bb16:  // pred: ^bb15
          %63 = memref.load %reinterpret_cast[%c1, %59] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %64 = vector.insert %63, %60 [%59] : f32 into vector<1xf32>
          cf.br ^bb18(%64 : vector<1xf32>)
        ^bb17:  // pred: ^bb15
          cf.br ^bb18(%60 : vector<1xf32>)
        ^bb18(%65: vector<1xf32>):  // 2 preds: ^bb16, ^bb17
          cf.br ^bb19
        ^bb19:  // pred: ^bb18
          %66 = arith.addi %59, %c1 : index
          cf.br ^bb14(%66, %65 : index, vector<1xf32>)
        ^bb20:  // pred: ^bb14
          %67 = vector.insert %60, %57 [1] : vector<1xf32> into vector<16x1xf32>
          %68 = vector.extract %47[2] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb21(%c0, %3 : index, vector<1xf32>)
        ^bb21(%69: index, %70: vector<1xf32>):  // 2 preds: ^bb20, ^bb26
          %71 = arith.cmpi slt, %69, %c1 : index
          cf.cond_br %71, ^bb22, ^bb27
        ^bb22:  // pred: ^bb21
          %72 = vector.extract %68[%69] : i1 from vector<1xi1>
          cf.cond_br %72, ^bb23, ^bb24
        ^bb23:  // pred: ^bb22
          %73 = memref.load %reinterpret_cast[%c2, %69] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %74 = vector.insert %73, %70 [%69] : f32 into vector<1xf32>
          cf.br ^bb25(%74 : vector<1xf32>)
        ^bb24:  // pred: ^bb22
          cf.br ^bb25(%70 : vector<1xf32>)
        ^bb25(%75: vector<1xf32>):  // 2 preds: ^bb23, ^bb24
          cf.br ^bb26
        ^bb26:  // pred: ^bb25
          %76 = arith.addi %69, %c1 : index
          cf.br ^bb21(%76, %75 : index, vector<1xf32>)
        ^bb27:  // pred: ^bb21
          %77 = vector.insert %70, %67 [2] : vector<1xf32> into vector<16x1xf32>
          %78 = vector.extract %47[3] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb28(%c0, %3 : index, vector<1xf32>)
        ^bb28(%79: index, %80: vector<1xf32>):  // 2 preds: ^bb27, ^bb33
          %81 = arith.cmpi slt, %79, %c1 : index
          cf.cond_br %81, ^bb29, ^bb34
        ^bb29:  // pred: ^bb28
          %82 = vector.extract %78[%79] : i1 from vector<1xi1>
          cf.cond_br %82, ^bb30, ^bb31
        ^bb30:  // pred: ^bb29
          %83 = memref.load %reinterpret_cast[%c3, %79] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %84 = vector.insert %83, %80 [%79] : f32 into vector<1xf32>
          cf.br ^bb32(%84 : vector<1xf32>)
        ^bb31:  // pred: ^bb29
          cf.br ^bb32(%80 : vector<1xf32>)
        ^bb32(%85: vector<1xf32>):  // 2 preds: ^bb30, ^bb31
          cf.br ^bb33
        ^bb33:  // pred: ^bb32
          %86 = arith.addi %79, %c1 : index
          cf.br ^bb28(%86, %85 : index, vector<1xf32>)
        ^bb34:  // pred: ^bb28
          %87 = vector.insert %80, %77 [3] : vector<1xf32> into vector<16x1xf32>
          %88 = vector.extract %47[4] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb35(%c0, %3 : index, vector<1xf32>)
        ^bb35(%89: index, %90: vector<1xf32>):  // 2 preds: ^bb34, ^bb40
          %91 = arith.cmpi slt, %89, %c1 : index
          cf.cond_br %91, ^bb36, ^bb41
        ^bb36:  // pred: ^bb35
          %92 = vector.extract %88[%89] : i1 from vector<1xi1>
          cf.cond_br %92, ^bb37, ^bb38
        ^bb37:  // pred: ^bb36
          %93 = memref.load %reinterpret_cast[%c4, %89] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %94 = vector.insert %93, %90 [%89] : f32 into vector<1xf32>
          cf.br ^bb39(%94 : vector<1xf32>)
        ^bb38:  // pred: ^bb36
          cf.br ^bb39(%90 : vector<1xf32>)
        ^bb39(%95: vector<1xf32>):  // 2 preds: ^bb37, ^bb38
          cf.br ^bb40
        ^bb40:  // pred: ^bb39
          %96 = arith.addi %89, %c1 : index
          cf.br ^bb35(%96, %95 : index, vector<1xf32>)
        ^bb41:  // pred: ^bb35
          %97 = vector.insert %90, %87 [4] : vector<1xf32> into vector<16x1xf32>
          %98 = vector.extract %47[5] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb42(%c0, %3 : index, vector<1xf32>)
        ^bb42(%99: index, %100: vector<1xf32>):  // 2 preds: ^bb41, ^bb47
          %101 = arith.cmpi slt, %99, %c1 : index
          cf.cond_br %101, ^bb43, ^bb48
        ^bb43:  // pred: ^bb42
          %102 = vector.extract %98[%99] : i1 from vector<1xi1>
          cf.cond_br %102, ^bb44, ^bb45
        ^bb44:  // pred: ^bb43
          %103 = memref.load %reinterpret_cast[%c5, %99] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %104 = vector.insert %103, %100 [%99] : f32 into vector<1xf32>
          cf.br ^bb46(%104 : vector<1xf32>)
        ^bb45:  // pred: ^bb43
          cf.br ^bb46(%100 : vector<1xf32>)
        ^bb46(%105: vector<1xf32>):  // 2 preds: ^bb44, ^bb45
          cf.br ^bb47
        ^bb47:  // pred: ^bb46
          %106 = arith.addi %99, %c1 : index
          cf.br ^bb42(%106, %105 : index, vector<1xf32>)
        ^bb48:  // pred: ^bb42
          %107 = vector.insert %100, %97 [5] : vector<1xf32> into vector<16x1xf32>
          %108 = vector.extract %47[6] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb49(%c0, %3 : index, vector<1xf32>)
        ^bb49(%109: index, %110: vector<1xf32>):  // 2 preds: ^bb48, ^bb54
          %111 = arith.cmpi slt, %109, %c1 : index
          cf.cond_br %111, ^bb50, ^bb55
        ^bb50:  // pred: ^bb49
          %112 = vector.extract %108[%109] : i1 from vector<1xi1>
          cf.cond_br %112, ^bb51, ^bb52
        ^bb51:  // pred: ^bb50
          %113 = memref.load %reinterpret_cast[%c6, %109] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %114 = vector.insert %113, %110 [%109] : f32 into vector<1xf32>
          cf.br ^bb53(%114 : vector<1xf32>)
        ^bb52:  // pred: ^bb50
          cf.br ^bb53(%110 : vector<1xf32>)
        ^bb53(%115: vector<1xf32>):  // 2 preds: ^bb51, ^bb52
          cf.br ^bb54
        ^bb54:  // pred: ^bb53
          %116 = arith.addi %109, %c1 : index
          cf.br ^bb49(%116, %115 : index, vector<1xf32>)
        ^bb55:  // pred: ^bb49
          %117 = vector.insert %110, %107 [6] : vector<1xf32> into vector<16x1xf32>
          %118 = vector.extract %47[7] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb56(%c0, %3 : index, vector<1xf32>)
        ^bb56(%119: index, %120: vector<1xf32>):  // 2 preds: ^bb55, ^bb61
          %121 = arith.cmpi slt, %119, %c1 : index
          cf.cond_br %121, ^bb57, ^bb62
        ^bb57:  // pred: ^bb56
          %122 = vector.extract %118[%119] : i1 from vector<1xi1>
          cf.cond_br %122, ^bb58, ^bb59
        ^bb58:  // pred: ^bb57
          %123 = memref.load %reinterpret_cast[%c7, %119] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %124 = vector.insert %123, %120 [%119] : f32 into vector<1xf32>
          cf.br ^bb60(%124 : vector<1xf32>)
        ^bb59:  // pred: ^bb57
          cf.br ^bb60(%120 : vector<1xf32>)
        ^bb60(%125: vector<1xf32>):  // 2 preds: ^bb58, ^bb59
          cf.br ^bb61
        ^bb61:  // pred: ^bb60
          %126 = arith.addi %119, %c1 : index
          cf.br ^bb56(%126, %125 : index, vector<1xf32>)
        ^bb62:  // pred: ^bb56
          %127 = vector.insert %120, %117 [7] : vector<1xf32> into vector<16x1xf32>
          %128 = vector.extract %47[8] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb63(%c0, %3 : index, vector<1xf32>)
        ^bb63(%129: index, %130: vector<1xf32>):  // 2 preds: ^bb62, ^bb68
          %131 = arith.cmpi slt, %129, %c1 : index
          cf.cond_br %131, ^bb64, ^bb69
        ^bb64:  // pred: ^bb63
          %132 = vector.extract %128[%129] : i1 from vector<1xi1>
          cf.cond_br %132, ^bb65, ^bb66
        ^bb65:  // pred: ^bb64
          %133 = memref.load %reinterpret_cast[%c8, %129] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %134 = vector.insert %133, %130 [%129] : f32 into vector<1xf32>
          cf.br ^bb67(%134 : vector<1xf32>)
        ^bb66:  // pred: ^bb64
          cf.br ^bb67(%130 : vector<1xf32>)
        ^bb67(%135: vector<1xf32>):  // 2 preds: ^bb65, ^bb66
          cf.br ^bb68
        ^bb68:  // pred: ^bb67
          %136 = arith.addi %129, %c1 : index
          cf.br ^bb63(%136, %135 : index, vector<1xf32>)
        ^bb69:  // pred: ^bb63
          %137 = vector.insert %130, %127 [8] : vector<1xf32> into vector<16x1xf32>
          %138 = vector.extract %47[9] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb70(%c0, %3 : index, vector<1xf32>)
        ^bb70(%139: index, %140: vector<1xf32>):  // 2 preds: ^bb69, ^bb75
          %141 = arith.cmpi slt, %139, %c1 : index
          cf.cond_br %141, ^bb71, ^bb76
        ^bb71:  // pred: ^bb70
          %142 = vector.extract %138[%139] : i1 from vector<1xi1>
          cf.cond_br %142, ^bb72, ^bb73
        ^bb72:  // pred: ^bb71
          %143 = memref.load %reinterpret_cast[%c9, %139] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %144 = vector.insert %143, %140 [%139] : f32 into vector<1xf32>
          cf.br ^bb74(%144 : vector<1xf32>)
        ^bb73:  // pred: ^bb71
          cf.br ^bb74(%140 : vector<1xf32>)
        ^bb74(%145: vector<1xf32>):  // 2 preds: ^bb72, ^bb73
          cf.br ^bb75
        ^bb75:  // pred: ^bb74
          %146 = arith.addi %139, %c1 : index
          cf.br ^bb70(%146, %145 : index, vector<1xf32>)
        ^bb76:  // pred: ^bb70
          %147 = vector.insert %140, %137 [9] : vector<1xf32> into vector<16x1xf32>
          %148 = vector.extract %47[10] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb77(%c0, %3 : index, vector<1xf32>)
        ^bb77(%149: index, %150: vector<1xf32>):  // 2 preds: ^bb76, ^bb82
          %151 = arith.cmpi slt, %149, %c1 : index
          cf.cond_br %151, ^bb78, ^bb83
        ^bb78:  // pred: ^bb77
          %152 = vector.extract %148[%149] : i1 from vector<1xi1>
          cf.cond_br %152, ^bb79, ^bb80
        ^bb79:  // pred: ^bb78
          %153 = memref.load %reinterpret_cast[%c10, %149] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %154 = vector.insert %153, %150 [%149] : f32 into vector<1xf32>
          cf.br ^bb81(%154 : vector<1xf32>)
        ^bb80:  // pred: ^bb78
          cf.br ^bb81(%150 : vector<1xf32>)
        ^bb81(%155: vector<1xf32>):  // 2 preds: ^bb79, ^bb80
          cf.br ^bb82
        ^bb82:  // pred: ^bb81
          %156 = arith.addi %149, %c1 : index
          cf.br ^bb77(%156, %155 : index, vector<1xf32>)
        ^bb83:  // pred: ^bb77
          %157 = vector.insert %150, %147 [10] : vector<1xf32> into vector<16x1xf32>
          %158 = vector.extract %47[11] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb84(%c0, %3 : index, vector<1xf32>)
        ^bb84(%159: index, %160: vector<1xf32>):  // 2 preds: ^bb83, ^bb89
          %161 = arith.cmpi slt, %159, %c1 : index
          cf.cond_br %161, ^bb85, ^bb90
        ^bb85:  // pred: ^bb84
          %162 = vector.extract %158[%159] : i1 from vector<1xi1>
          cf.cond_br %162, ^bb86, ^bb87
        ^bb86:  // pred: ^bb85
          %163 = memref.load %reinterpret_cast[%c11, %159] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %164 = vector.insert %163, %160 [%159] : f32 into vector<1xf32>
          cf.br ^bb88(%164 : vector<1xf32>)
        ^bb87:  // pred: ^bb85
          cf.br ^bb88(%160 : vector<1xf32>)
        ^bb88(%165: vector<1xf32>):  // 2 preds: ^bb86, ^bb87
          cf.br ^bb89
        ^bb89:  // pred: ^bb88
          %166 = arith.addi %159, %c1 : index
          cf.br ^bb84(%166, %165 : index, vector<1xf32>)
        ^bb90:  // pred: ^bb84
          %167 = vector.insert %160, %157 [11] : vector<1xf32> into vector<16x1xf32>
          %168 = vector.extract %47[12] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb91(%c0, %3 : index, vector<1xf32>)
        ^bb91(%169: index, %170: vector<1xf32>):  // 2 preds: ^bb90, ^bb96
          %171 = arith.cmpi slt, %169, %c1 : index
          cf.cond_br %171, ^bb92, ^bb97
        ^bb92:  // pred: ^bb91
          %172 = vector.extract %168[%169] : i1 from vector<1xi1>
          cf.cond_br %172, ^bb93, ^bb94
        ^bb93:  // pred: ^bb92
          %173 = memref.load %reinterpret_cast[%c12, %169] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %174 = vector.insert %173, %170 [%169] : f32 into vector<1xf32>
          cf.br ^bb95(%174 : vector<1xf32>)
        ^bb94:  // pred: ^bb92
          cf.br ^bb95(%170 : vector<1xf32>)
        ^bb95(%175: vector<1xf32>):  // 2 preds: ^bb93, ^bb94
          cf.br ^bb96
        ^bb96:  // pred: ^bb95
          %176 = arith.addi %169, %c1 : index
          cf.br ^bb91(%176, %175 : index, vector<1xf32>)
        ^bb97:  // pred: ^bb91
          %177 = vector.insert %170, %167 [12] : vector<1xf32> into vector<16x1xf32>
          %178 = vector.extract %47[13] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb98(%c0, %3 : index, vector<1xf32>)
        ^bb98(%179: index, %180: vector<1xf32>):  // 2 preds: ^bb97, ^bb103
          %181 = arith.cmpi slt, %179, %c1 : index
          cf.cond_br %181, ^bb99, ^bb104
        ^bb99:  // pred: ^bb98
          %182 = vector.extract %178[%179] : i1 from vector<1xi1>
          cf.cond_br %182, ^bb100, ^bb101
        ^bb100:  // pred: ^bb99
          %183 = memref.load %reinterpret_cast[%c13, %179] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %184 = vector.insert %183, %180 [%179] : f32 into vector<1xf32>
          cf.br ^bb102(%184 : vector<1xf32>)
        ^bb101:  // pred: ^bb99
          cf.br ^bb102(%180 : vector<1xf32>)
        ^bb102(%185: vector<1xf32>):  // 2 preds: ^bb100, ^bb101
          cf.br ^bb103
        ^bb103:  // pred: ^bb102
          %186 = arith.addi %179, %c1 : index
          cf.br ^bb98(%186, %185 : index, vector<1xf32>)
        ^bb104:  // pred: ^bb98
          %187 = vector.insert %180, %177 [13] : vector<1xf32> into vector<16x1xf32>
          %188 = vector.extract %47[14] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb105(%c0, %3 : index, vector<1xf32>)
        ^bb105(%189: index, %190: vector<1xf32>):  // 2 preds: ^bb104, ^bb110
          %191 = arith.cmpi slt, %189, %c1 : index
          cf.cond_br %191, ^bb106, ^bb111
        ^bb106:  // pred: ^bb105
          %192 = vector.extract %188[%189] : i1 from vector<1xi1>
          cf.cond_br %192, ^bb107, ^bb108
        ^bb107:  // pred: ^bb106
          %193 = memref.load %reinterpret_cast[%c14, %189] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %194 = vector.insert %193, %190 [%189] : f32 into vector<1xf32>
          cf.br ^bb109(%194 : vector<1xf32>)
        ^bb108:  // pred: ^bb106
          cf.br ^bb109(%190 : vector<1xf32>)
        ^bb109(%195: vector<1xf32>):  // 2 preds: ^bb107, ^bb108
          cf.br ^bb110
        ^bb110:  // pred: ^bb109
          %196 = arith.addi %189, %c1 : index
          cf.br ^bb105(%196, %195 : index, vector<1xf32>)
        ^bb111:  // pred: ^bb105
          %197 = vector.insert %190, %187 [14] : vector<1xf32> into vector<16x1xf32>
          %198 = vector.extract %47[15] : vector<1xi1> from vector<16x1xi1>
          cf.br ^bb112(%c0, %3 : index, vector<1xf32>)
        ^bb112(%199: index, %200: vector<1xf32>):  // 2 preds: ^bb111, ^bb117
          %201 = arith.cmpi slt, %199, %c1 : index
          cf.cond_br %201, ^bb113, ^bb118
        ^bb113:  // pred: ^bb112
          %202 = vector.extract %198[%199] : i1 from vector<1xi1>
          cf.cond_br %202, ^bb114, ^bb115
        ^bb114:  // pred: ^bb113
          %203 = memref.load %reinterpret_cast[%c15, %199] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %204 = vector.insert %203, %200 [%199] : f32 into vector<1xf32>
          cf.br ^bb116(%204 : vector<1xf32>)
        ^bb115:  // pred: ^bb113
          cf.br ^bb116(%200 : vector<1xf32>)
        ^bb116(%205: vector<1xf32>):  // 2 preds: ^bb114, ^bb115
          cf.br ^bb117
        ^bb117:  // pred: ^bb116
          %206 = arith.addi %199, %c1 : index
          cf.br ^bb112(%206, %205 : index, vector<1xf32>)
        ^bb118:  // pred: ^bb112
          %207 = vector.insert %200, %197 [15] : vector<1xf32> into vector<16x1xf32>
          %208 = vector.broadcast %207 : vector<16x1xf32> to vector<8x16x1xf32>
          %209 = vector.transpose %208, [1, 0, 2] : vector<8x16x1xf32> to vector<16x8x1xf32>
          %210 = vector.create_mask %18, %26 : vector<1x8xi1>
          %211 = vector.extract %210[0] : vector<8xi1> from vector<1x8xi1>
          cf.br ^bb119(%c0, %1 : index, vector<8xf32>)
        ^bb119(%212: index, %213: vector<8xf32>):  // 2 preds: ^bb118, ^bb124
          %214 = arith.cmpi slt, %212, %c8 : index
          cf.cond_br %214, ^bb120, ^bb125
        ^bb120:  // pred: ^bb119
          %215 = vector.extract %211[%212] : i1 from vector<8xi1>
          cf.cond_br %215, ^bb121, ^bb122
        ^bb121:  // pred: ^bb120
          %216 = memref.load %reinterpret_cast_4[%c0, %212] : memref<?x?xf32, strided<[?, ?], offset: ?>>
          %217 = vector.insert %216, %213 [%212] : f32 into vector<8xf32>
          cf.br ^bb123(%217 : vector<8xf32>)
        ^bb122:  // pred: ^bb120
          cf.br ^bb123(%213 : vector<8xf32>)
        ^bb123(%218: vector<8xf32>):  // 2 preds: ^bb121, ^bb122
          cf.br ^bb124
        ^bb124:  // pred: ^bb123
          %219 = arith.addi %212, %c1 : index
          cf.br ^bb119(%219, %218 : index, vector<8xf32>)
        ^bb125:  // pred: ^bb119
          %220 = vector.insert %213, %2 [0] : vector<8xf32> into vector<1x8xf32>
          %221 = vector.broadcast %220 : vector<1x8xf32> to vector<16x1x8xf32>
          %222 = vector.transpose %221, [0, 2, 1] : vector<16x1x8xf32> to vector<16x8x1xf32>
          %223 = vector.create_mask %24, %26 : vector<16x8xi1>
          %224 = vector.extract %223[0] : vector<8xi1> from vector<16x8xi1>
          %225 = vector.transfer_read %reinterpret_cast_5[%c0, %c0], %5, %224 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %226 = vector.insert %225, %0 [0] : vector<8xf32> into vector<16x8xf32>
          %227 = vector.extract %223[1] : vector<8xi1> from vector<16x8xi1>
          %228 = vector.transfer_read %reinterpret_cast_5[%c1, %c0], %5, %227 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %229 = vector.insert %228, %226 [1] : vector<8xf32> into vector<16x8xf32>
          %230 = vector.extract %223[2] : vector<8xi1> from vector<16x8xi1>
          %231 = vector.transfer_read %reinterpret_cast_5[%c2, %c0], %5, %230 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %232 = vector.insert %231, %229 [2] : vector<8xf32> into vector<16x8xf32>
          %233 = vector.extract %223[3] : vector<8xi1> from vector<16x8xi1>
          %234 = vector.transfer_read %reinterpret_cast_5[%c3, %c0], %5, %233 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %235 = vector.insert %234, %232 [3] : vector<8xf32> into vector<16x8xf32>
          %236 = vector.extract %223[4] : vector<8xi1> from vector<16x8xi1>
          %237 = vector.transfer_read %reinterpret_cast_5[%c4, %c0], %5, %236 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %238 = vector.insert %237, %235 [4] : vector<8xf32> into vector<16x8xf32>
          %239 = vector.extract %223[5] : vector<8xi1> from vector<16x8xi1>
          %240 = vector.transfer_read %reinterpret_cast_5[%c5, %c0], %5, %239 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %241 = vector.insert %240, %238 [5] : vector<8xf32> into vector<16x8xf32>
          %242 = vector.extract %223[6] : vector<8xi1> from vector<16x8xi1>
          %243 = vector.transfer_read %reinterpret_cast_5[%c6, %c0], %5, %242 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %244 = vector.insert %243, %241 [6] : vector<8xf32> into vector<16x8xf32>
          %245 = vector.extract %223[7] : vector<8xi1> from vector<16x8xi1>
          %246 = vector.transfer_read %reinterpret_cast_5[%c7, %c0], %5, %245 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %247 = vector.insert %246, %244 [7] : vector<8xf32> into vector<16x8xf32>
          %248 = vector.extract %223[8] : vector<8xi1> from vector<16x8xi1>
          %249 = vector.transfer_read %reinterpret_cast_5[%c8, %c0], %5, %248 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %250 = vector.insert %249, %247 [8] : vector<8xf32> into vector<16x8xf32>
          %251 = vector.extract %223[9] : vector<8xi1> from vector<16x8xi1>
          %252 = vector.transfer_read %reinterpret_cast_5[%c9, %c0], %5, %251 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %253 = vector.insert %252, %250 [9] : vector<8xf32> into vector<16x8xf32>
          %254 = vector.extract %223[10] : vector<8xi1> from vector<16x8xi1>
          %255 = vector.transfer_read %reinterpret_cast_5[%c10, %c0], %5, %254 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %256 = vector.insert %255, %253 [10] : vector<8xf32> into vector<16x8xf32>
          %257 = vector.extract %223[11] : vector<8xi1> from vector<16x8xi1>
          %258 = vector.transfer_read %reinterpret_cast_5[%c11, %c0], %5, %257 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %259 = vector.insert %258, %256 [11] : vector<8xf32> into vector<16x8xf32>
          %260 = vector.extract %223[12] : vector<8xi1> from vector<16x8xi1>
          %261 = vector.transfer_read %reinterpret_cast_5[%c12, %c0], %5, %260 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %262 = vector.insert %261, %259 [12] : vector<8xf32> into vector<16x8xf32>
          %263 = vector.extract %223[13] : vector<8xi1> from vector<16x8xi1>
          %264 = vector.transfer_read %reinterpret_cast_5[%c13, %c0], %5, %263 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %265 = vector.insert %264, %262 [13] : vector<8xf32> into vector<16x8xf32>
          %266 = vector.extract %223[14] : vector<8xi1> from vector<16x8xi1>
          %267 = vector.transfer_read %reinterpret_cast_5[%c14, %c0], %5, %266 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %268 = vector.insert %267, %265 [14] : vector<8xf32> into vector<16x8xf32>
          %269 = vector.extract %223[15] : vector<8xi1> from vector<16x8xi1>
          %270 = vector.transfer_read %reinterpret_cast_5[%c15, %c0], %5, %269 {in_bounds = [true]} : memref<?x?xf32, strided<[4096, 1], offset: ?>>, vector<8xf32>
          %271 = vector.insert %270, %268 [15] : vector<8xf32> into vector<16x8xf32>
          %272 = arith.mulf %209, %222 : vector<16x8x1xf32>
          %273 = vector.create_mask %24, %26, %18 : vector<16x8x1xi1>
          %274 = vector.shape_cast %273 : vector<16x8x1xi1> to vector<16x8xi1>
          %275 = vector.shape_cast %272 : vector<16x8x1xf32> to vector<16x8xf32>
          %276 = arith.addf %271, %275 : vector<16x8xf32>
          %277 = arith.select %274, %276, %275 : vector<16x8xi1>, vector<16x8xf32>
          %278 = vector.extract %277[0] : vector<8xf32> from vector<16x8xf32>
          %279 = vector.extract %223[0] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %278, %reinterpret_cast_5[%c0, %c0], %279 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %280 = vector.extract %277[1] : vector<8xf32> from vector<16x8xf32>
          %281 = vector.extract %223[1] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %280, %reinterpret_cast_5[%c1, %c0], %281 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %282 = vector.extract %277[2] : vector<8xf32> from vector<16x8xf32>
          %283 = vector.extract %223[2] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %282, %reinterpret_cast_5[%c2, %c0], %283 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %284 = vector.extract %277[3] : vector<8xf32> from vector<16x8xf32>
          %285 = vector.extract %223[3] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %284, %reinterpret_cast_5[%c3, %c0], %285 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %286 = vector.extract %277[4] : vector<8xf32> from vector<16x8xf32>
          %287 = vector.extract %223[4] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %286, %reinterpret_cast_5[%c4, %c0], %287 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %288 = vector.extract %277[5] : vector<8xf32> from vector<16x8xf32>
          %289 = vector.extract %223[5] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %288, %reinterpret_cast_5[%c5, %c0], %289 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %290 = vector.extract %277[6] : vector<8xf32> from vector<16x8xf32>
          %291 = vector.extract %223[6] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %290, %reinterpret_cast_5[%c6, %c0], %291 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %292 = vector.extract %277[7] : vector<8xf32> from vector<16x8xf32>
          %293 = vector.extract %223[7] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %292, %reinterpret_cast_5[%c7, %c0], %293 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %294 = vector.extract %277[8] : vector<8xf32> from vector<16x8xf32>
          %295 = vector.extract %223[8] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %294, %reinterpret_cast_5[%c8, %c0], %295 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %296 = vector.extract %277[9] : vector<8xf32> from vector<16x8xf32>
          %297 = vector.extract %223[9] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %296, %reinterpret_cast_5[%c9, %c0], %297 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %298 = vector.extract %277[10] : vector<8xf32> from vector<16x8xf32>
          %299 = vector.extract %223[10] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %298, %reinterpret_cast_5[%c10, %c0], %299 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %300 = vector.extract %277[11] : vector<8xf32> from vector<16x8xf32>
          %301 = vector.extract %223[11] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %300, %reinterpret_cast_5[%c11, %c0], %301 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %302 = vector.extract %277[12] : vector<8xf32> from vector<16x8xf32>
          %303 = vector.extract %223[12] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %302, %reinterpret_cast_5[%c12, %c0], %303 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %304 = vector.extract %277[13] : vector<8xf32> from vector<16x8xf32>
          %305 = vector.extract %223[13] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %304, %reinterpret_cast_5[%c13, %c0], %305 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %306 = vector.extract %277[14] : vector<8xf32> from vector<16x8xf32>
          %307 = vector.extract %223[14] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %306, %reinterpret_cast_5[%c14, %c0], %307 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %308 = vector.extract %277[15] : vector<8xf32> from vector<16x8xf32>
          %309 = vector.extract %223[15] : vector<8xi1> from vector<16x8xi1>
          vector.transfer_write %308, %reinterpret_cast_5[%c15, %c0], %309 {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %310 = arith.addi %21, %c8 : index
          cf.br ^bb5(%310 : index)
        ^bb126:  // pred: ^bb5
          %311 = arith.addi %19, %c16 : index
          cf.br ^bb3(%311 : index)
        ^bb127:  // pred: ^bb3
          %312 = arith.addi %15, %c500 : index
          cf.br ^bb1(%312 : index)
        ^bb128:  // pred: ^bb1
          omp.yield
        }
      }
      omp.terminator
    }
    %7 = call @get_time() : () -> i64
    %8 = arith.subi %7, %6 : i64
    call @print_gflops(%c4096_i64, %c4096_i64, %c4096_i64, %8) : (i64, i64, i64, i64) -> ()
    return %alloc : memref<4096x4096xf32>
  }
  func.func @main() -> i32 {
    %cst = arith.constant 1.000000e-03 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 8.192000e+03 : f32
    %0 = memref.get_global @__constant_4096x4096xf32 : memref<4096x4096xf32>
    %1 = memref.get_global @__constant_4096x4096xf32_0 : memref<4096x4096xf32>
    %cast = memref.cast %0 : memref<4096x4096xf32> to memref<4096x4096xf32, strided<[?, ?], offset: ?>>
    %cast_1 = memref.cast %1 : memref<4096x4096xf32> to memref<4096x4096xf32, strided<[?, ?], offset: ?>>
    %2 = call @core_op(%cast, %cast_1) : (memref<4096x4096xf32, strided<[?, ?], offset: ?>>, memref<4096x4096xf32, strided<[?, ?], offset: ?>>) -> memref<4096x4096xf32>
    %3 = memref.load %2[%c0, %c0] : memref<4096x4096xf32>
    %4 = arith.subf %3, %cst_0 : f32
    %5 = math.absf %4 : f32
    %6 = arith.cmpf oge, %5, %cst : f32
    %7 = arith.extui %6 : i1 to i32
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %2 : memref<4096x4096xf32> -> memref<f32>, index, index, index, index, index
    memref.dealloc %base_buffer : memref<f32>
    return %7 : i32
  }
}

