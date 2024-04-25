# Effcient_vmamba_block
# mamba 在windows 上配置
# 安装Mamba_ssm
## Pytorch

要求：cuda11.8 

直接在官网上面https://pytorch.org/get-started/locally/按照下面的pip语句（蓝色划线处）在命令行安装即可。
![image](https://github.com/Jacky-Android/Effcient_vmamba_block/assets/55181594/30979539-158d-4e1a-9c6b-fc4391ef1651)


[教程](https://jade-alphabet-c76.notion.site/Win-Mamba-89338d8b2ffe4600b67047cd6d919afd)


# vmamba block that runs efficiently on windows

Windows配置的mamba 不能使用selective_scan_cuda，导致需要for循环，而官方实现执行了更快的并行扫描，另外还具有硬件感知能力（如 FlashAttention），在win上艰难执行
修改为
```python
def selective_scan(self, u, delta, A, B, C, D):
      
      
        (b, l, d) = u.shape #(B,L,D)
        n = A.shape[1] # N
        
        # einsum 操作实际上是将 delta 和 A 张量的最后两个维度进行矩阵乘法，并在前面添加了两个维度（b 和 l）
        # torch.exp 函数对这个张量中的每个元素进行指数化，即将每个元素取指数值。
        deltaA = torch.exp(einsum(delta, A,'b l d, d n -> b l d n'))   # (B,L,D) * (D,N) -> (B,L,D,N)
        # 将 delta、B 和 u 张量的对应位置元素相乘，并在最后一个维度上进行求和，输出一个新的张量。
        g = B.shape[1]
        B = repeat(B, "b g n l -> b l (g H) n", H=A.shape[0] // B.shape[1])
        
        deltaB_u = einsum(delta, B, u, 'b l d, b l d n, b l d -> b l d n')  # (B,L,D)*(B,L,N)*(B,L,D)->(B,L,D,N)
        
        '''
        执行 selective scan (see scan_SSM() in The Annotated S4 [2])
        # 注意，下面的代码是顺序执行的, 然而在官方代码中使用更快的并行扫描算法实现的(类似于FlashAttention，采用硬件感知扫描)。
        '''
        x = torch.zeros((b, d, n), device=deltaA.device)
        C = repeat(C, "B G N L -> B (G H) N L", H=A.shape[0] // C.shape[1])
        ys = []  
          
        '''for i in range(l):  # 这里使用for循环的方式只用来说明核心的逻辑，原代码中采用并行扫描算法
            x = deltaA[:, i] * x + deltaB_u[:, i] # x(t + 1) = Ax(t) + Bu(t)
            #print(x.shape,C[:, :, :, i].shape)
            y = einsum(x, C[:, :, :, i], 'b d n, b d n -> b d') # y(t) = Cx(t)  (B,D,N)*(B,N)->(B,D) 
            ys.append(y)
        y = torch.stack(ys, dim=1)  # 大小 (b, l, d_in)  (B,L,D)'''
        x_list = []
        '''for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]  # x(t + 1) = Ax(t) + Bu(t)
            x_list.append(x.unsqueeze(1))  # 添加到列表中，稍后进行拼接

        x_concat = torch.cat(x_list, dim=1)  # 在第二维度上拼接所有的 x'''
        #print(x_concat.shape,u.shape,C[:, :, :, i].shape)
        x_concat = deltaA * x.unsqueeze(1) + deltaB_u 
        #print(x_concat.shape,u.shape,C[:, :, :, l-1].shape)
        y = einsum( x_concat, C[:, :, :, l-1],'b l d n,b d n->b l d') 
        
        
       
        y = y + u * D # y(t) = Cx(t)+Du(t)
        y = y.permute(0,2,1)
        return y #(B,L,D)
```

