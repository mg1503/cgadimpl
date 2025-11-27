func.func @main(%input1: tensor<4096x4096xi32>, 
                %input2: tensor<4096x4096xi32>,
                %input3: tensor<4096xi32>) -> tensor<4096x4096xi32> {
  %A = nova.constant {value = dense<[1]> : tensor<1xi32>} : tensor<1xi32>
  %B = nova.matmul %input1, %input2 : tensor<4096x4096xi32>, tensor<4096x4096xi32>
  %C = nova.sub %B, %input3 : tensor<4096x4096xi32>, tensor<4096xi32>
  %D= nova.add %B, %C: tensor<4096x4096xi32>, tensor<4096x4096xi32>
  %E = nova.mul %D, %C: tensor<4096x4096xi32>, tensor<4096x4096xi32>
  %F = nova.pow %D, %A: tensor<4096x4096xi32>, tensor<1xi32>
  %G = nova.sin  %E: tensor<4096x4096xi32>
  %H = nova.abs  %F: tensor<4096x4096xi32>
  %I = nova.relu %G:tensor<4096x4096xi32>

  %J=nova.div %H,%I:tensor<4096x4096xi32>,tensor<4096x4096xi32>

  %K=nova.mod %I,%J:tensor<4096x4096xi32>,tensor<4096x4096xi32>
  
  %L=nova.max %J,%K:tensor<4096x4096xi32>,tensor<4096x4096xi32>
  %M=nova.min %K,%L:tensor<4096x4096xi32>,tensor<4096x4096xi32>
  %N=nova.neg %M:tensor<4096x4096xi32>

  return %N : tensor<4096x4096xi32>
}
// convert-nova-to-tosa --convert-nova-to-linalg --convert-nova-to-arith 